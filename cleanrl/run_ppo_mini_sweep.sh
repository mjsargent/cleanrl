for i in 1,2,3,4,5 
do 
	python ppo_minigrid.py --prod-mode 1 --cuda 1 --capture-video --gym-id MiniGrid-SimpleCrossingS9N3-v0 --total-timesteps 2500000 --num-steps 32
	python ppo_levy_minigrid.py --prod-mode 1 --cuda 1 --capture-video --gym-id MiniGrid-SimpleCrossingS9N3-v0 --total-timesteps 2500000 --num-steps 32
	CUDA_LAUNCH_BLOCKING=1 python ppo_levy_minigrid.py --prod-mode 1 --cuda 1 --capture-video --gym-id MiniGrid-SimpleCrossingS9N3-v0 --total-timesteps 2500000 --num-steps 31
done
	
