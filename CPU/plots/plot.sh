#!/bin/bash
postfix="2"

# Create directories with postfix
mkdir -p out${postfix}/paper${postfix}
mkdir -p out${postfix}/paper_pre${postfix}
mkdir -p out${postfix}/simulations${postfix}

# Generate plots from benchmark data
python3 plot.py

# Move and rename figures with postfix
mv out/paper_pre/unidirectional_bw_stripe.pdf out${postfix}/paper${postfix}/fig1${postfix}.pdf
mv out/paper_pre/unidirectional_bw_conc.pdf out${postfix}/paper${postfix}/fig2${postfix}.pdf
mv out/paper_pre/lat_bw.pdf out${postfix}/paper${postfix}/fig3${postfix}.pdf
mv out/paper_pre/uni_vs_bi.pdf out${postfix}/paper${postfix}/fig4${postfix}.pdf
mv out/paper_pre/unidirectional_bw_instances.pdf out${postfix}/paper${postfix}/fig5${postfix}.pdf
mv out/paper_pre/hoverboard.pdf out${postfix}/paper${postfix}/fig6${postfix}.pdf
mv out/paper_pre/os_noise_instance_type_long.pdf out${postfix}/paper${postfix}/fig9${postfix}.pdf
mv out/paper_pre/noise_lat_time_alloc_long.pdf out${postfix}/paper${postfix}/fig10${postfix}.pdf
mv out/paper_pre/noise_lat_instance_type_long.pdf out${postfix}/paper${postfix}/fig11${postfix}.pdf
mv out/paper_pre/noise_bw_time_alloc_long.pdf out${postfix}/paper${postfix}/fig12${postfix}.pdf
mv out/paper_pre/noise_bw_instance_type_long.pdf out${postfix}/paper${postfix}/fig13${postfix}.pdf


# Generate plots from simulation data
# ./plot_sim.py
# mv out/simulations/validation_relative.pdf out/paper/fig14.pdf
# mv out/simulations/scalability_nwaydissemination_16.pdf out/paper/fig15.pdf
# mv out/simulations/scalability_pipelinedring_536870912.pdf out/paper/fig16.pdf
# mv out/simulations/cost_on-demand_nwaydissemination_128_new.pdf out/paper/fig17.pdf
