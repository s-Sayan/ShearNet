ForkLens baseline on the fourth task: the non-equivariant two-branch model,
no response terms, same dataset and training budget as `fourth_inloop_control`.

Three arms span the two axes that matter here:

    fourth_inloop_forklens   equivariant: no    response: no
    fourth_inloop_control    equivariant: yes   response: no
    fourth_inloop_response   equivariant: yes   response: yes

A separate D4CNN arm is deliberately absent: on this task it would be the same
architecture as `fourth_inloop_control` (d4-fork-like) differing only in
hyperparameters, so it would mostly restate that column.
