from tqdm import tqdm
import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from shearnet.models import SimpleGalaxyNN

def save_model(state, save_path):
    """Save the trained model's parameters."""
    os.makedirs(save_path, exist_ok=True)
    jax.numpy.save(os.path.join(save_path, 'model_params.npy'), state.params)

def load_model(model, load_path):
    """Load the model's parameters."""
    params = jax.numpy.load(os.path.join(load_path, 'model_params.npy'), allow_pickle=True)
    return model, params

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
    preds = state.apply_fn(params, images)
    loss = optax.l2_loss(preds, labels).mean()
    return loss  # Mean Squared Error

@jax.jit
def train_step(state, images, labels):
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=False)
    loss, grads = grad_fn(state, state.params, images, labels)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_model(images, labels, rng_key, epochs=10, batch_size=32, save_path=None, model_name="my_model"):
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