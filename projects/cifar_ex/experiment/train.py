from projects.cifar_ex.runtime import CifarLab

lab = CifarLab("trial_run", batch_size=64, n_epochs=10)

lab.run()
