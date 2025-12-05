rm -rf Diagnostics/*
./ExaGOOP2d.gnu.ex inputs_elasticdisk.in
python3 plot_energy.py energy.png
open energy.png
