ForkLens baseline on the fourth task: the same two-branch topology with the
equivariance removed, no response terms, and the same dataset, seeds and
training budget as `fourth_shearnet_control`.

Three arms span the two axes that matter, each differing from the next in one
thing only:

    fourth_forklens           equivariant: no    response: no
    fourth_shearnet_control   equivariant: yes   response: no
    fourth_shearnet           equivariant: yes   response: yes

So `forklens -> control` prices the equivariance and `control -> shearnet`
prices the response terms, with no third change confounding either step.

A separate D4CNN arm is deliberately absent: on this task it would be the same
architecture as `fourth_shearnet_control` (`d4-fork-like`) differing only in
hyperparameters, so it would mostly restate that column. The D4CNN of Lin et
al. is a different comparison -- its network input is the PSF-homogenised
`f_h`, not a raw stamp -- and that belongs in a run that changes the input
contract, not the architecture.

    ./sub.sh fourth_forklens
