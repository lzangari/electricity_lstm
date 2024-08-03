cd C:/Repositories/PlotNeuralNet/pyexamples
bash ../tikzmake.sh test_elec_naive
bash ../tikzmake.sh test_elec_s2s
bash ../tikzmake.sh test_elec_s2s_reg
bash ../tikzmake.sh test_elec_stacked


# move all the the pdf starting with "test_elec" to the folder "models_figures"
mv test_elec_naive.pdf C:/Repositories/electricity_lstm/thesis_figures/model_figures
mv test_elec_s2s.pdf C:/Repositories/electricity_lstm/thesis_figures/model_figures
mv test_elec_s2s_reg.pdf C:/Repositories/electricity_lstm/thesis_figures/model_figures
mv test_elec_stacked.pdf C:/Repositories/electricity_lstm/thesis_figures/model_figures

