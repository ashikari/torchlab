from projects.cifar_ex.runtime import CifarLab

lab = CifarLab("trial_run", batch_size=5, n_epochs=5)

lab.run()