from tqdm import tqdm
import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from shearnet.models import *

def save_checkpoint(state, step, checkpoint_dir, model_name, overwrite=True):
    """Save the model checkpoint."""
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=step,
        prefix=model_name,
        overwrite=overwrite
    )
    print(f"Checkpoint saved at step {step}.")

def loss_fn(state, params, images, labels):
    preds, new_model_state = state.apply_fn(params, images, mutable=['batch_stats'], train=True)
    # preds = state.apply_fn(params, images)
    loss = optax.l2_loss(preds, labels).mean()
    return loss, new_model_state  # Mean Squared Error

@jax.jit
def train_step(state, images, labels):
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    (loss, new_model_state), grads = grad_fn(state, state.params, images, labels)
    state = state.apply_gradients(grads=grads)
    state.params["batch_stats"] = new_model_state["batch_stats"]
    return state, loss

@jax.jit
def eval_step(state, images, labels):
    preds = state.apply_fn(state.params, images, train=False) # mutable=False
    loss = optax.l2_loss(preds, labels).mean()
    # loss = loss_fn(state, state.params, images, labels)
    return loss

def train_model(images, labels, rng_key, epochs=10, batch_size=32, nn="simple", save_path=None, model_name="my_model"):
    if nn == "simple":
        model = SimpleGalaxyNN()  # Initialize the model
    elif nn == "enhanced":
        model = GalaxyResNet()  # Initialize the complex model
    elif nn == "vgg16":
        model = VGG16() # Initialize the transfer learning model
    else:
        raise ValueError("Invalid model type specified.")
    model = SimpleGalaxyNN()  # Initialize the model
    params = model.init(rng_key, jnp.ones_like(images[0]))  # Initialize model parameters
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    
    epoch_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Shuffle the data at the beginning of each epoch
        rng_key, subkey = jax.random.split(rng_key)
        perm = jax.random.permutation(subkey, len(images))  # Create a permutation of indices
        shuffled_images = images[perm]  # Apply the permutation to images
        shuffled_labels = labels[perm]  # Apply the same permutation to labels
        epoch_loss = 0
        count = 0

        with tqdm(total=len(images) // batch_size) as pbar:
            for i in range(0, len(images), batch_size):
                batch_images = shuffled_images[i:i + batch_size]
                batch_labels = shuffled_labels[i:i + batch_size]
                state, loss = train_step(state, batch_images, batch_labels)
                epoch_loss += loss
                count += 1
                pbar.update(1)
        print(f"Loss: {loss.item()}")
        epoch_loss /= count
        epoch_losses.append(epoch_loss)

        # Save the model after every epoch if a save path is provided
        if save_path:
            #save_model(state, save_path)
            save_checkpoint(state, step=epoch + 1, checkpoint_dir=save_path, model_name=model_name, overwrite=True)
    
    return state, epoch_losses

def train_modelv2(images, labels, rng_key, epochs=10, batch_size=32, nn="simple", save_path=None, model_name="my_model", val_split=0.2, eval_interval=1, patience=5, weight_decay=1e-4):
    # Split into train and validation sets
    split_idx = int(len(images) * (1 - val_split))
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    if nn == "simple":
        model = SimpleGalaxyNN()  # Initialize the model
    elif nn == "enhanced":
        model = EnhancedGalaxyNN()  # Initialize the complex model
    elif nn == "resnet":
        model = GalaxyResNet()  # Initialize the complex model
    elif nn == "vgg16":
        model = VGG16()   # Initialize the transfer learning model
    elif nn == "fork":
        model = ForkCNN()
    else:
        raise ValueError("Invalid model type specified.")
    
    params = model.init(rng_key, jnp.ones_like(images[0]))  # Initialize model parameters
    print(params.keys())
    lr_schedule = optax.cosine_decay_schedule(init_value=1e-4, decay_steps=epochs * (len(train_images) // batch_size))
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')  # Initialize best validation loss
    patience_counter = 0  # Counter for early stopping

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Shuffle training data
        rng_key, subkey = jax.random.split(rng_key)
        perm = jax.random.permutation(subkey, len(train_images))
        shuffled_train_images = train_images[perm]
        shuffled_train_labels = train_labels[perm]

        train_loss, total_samples, count = 0, 0, 0
        with tqdm(total=len(train_images) // batch_size) as pbar:
            for i in range(0, len(train_images), batch_size):
                batch_images = shuffled_train_images[i:i + batch_size]
                batch_labels = shuffled_train_labels[i:i + batch_size]
                # batch = {"input": batch_images, "labels": batch_labels}
                batch_size_actual = len(batch_images)
                state, loss = train_step(state, batch_images, batch_labels)
                train_loss += loss * batch_size_actual
                total_samples += batch_size_actual
                pbar.update(1)
        train_loss /= total_samples
        train_losses.append(train_loss)

        # Evaluate validation loss at specified intervals
        if (epoch + 1) % eval_interval == 0:
            val_loss, total_samples = 0, 0
            for i in range(0, len(val_images), batch_size):
                batch_images = val_images[i:i + batch_size]
                batch_labels = val_labels[i:i + batch_size]
                batch_size_actual = len(batch_images)
                loss = eval_step(state, batch_images, batch_labels)
                val_loss += loss * batch_size_actual
                total_samples += batch_size_actual
            val_loss /= total_samples
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4f}")

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss  # Update the best validation loss
                patience_counter = 0  # Reset patience counter
                if save_path:  # Save the best model
                    save_checkpoint(state, step=epoch + 1, checkpoint_dir=save_path, model_name=f"{model_name}_best", overwrite=True)
                print(f"New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Save checkpoint
        if save_path:
            save_checkpoint(state, step=epoch + 1, checkpoint_dir=save_path, model_name=model_name, overwrite=True)

    return state, train_losses, val_losses