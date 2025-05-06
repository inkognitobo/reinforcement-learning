# Reinforcement Learning

## Notes

- $TD(\lambda)$ return estimation: $V_{targ} = est_returns = adv + values$
- Observation normalisation and clipping can help performance
- Reward Scaling and clipping can help performance

## TODO:

Look into: 

- Clip range annealing
- Parallellized gradient updates
- Early stopping
- Invalid action masking

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347):
```bibtex
   @misc{schulman:2017:ppo,
   title={Proximal Policy Optimization Algorithms},
   author={John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov},
   year={2017},
   eprint={1707.06347},
   archivePrefix={arXiv},
   primaryClass={cs.LG},
   url={https://arxiv.org/abs/1707.06347},
}
```
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
