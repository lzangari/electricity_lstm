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


cd C:/Repositories/electricity_lstm/thesis_figures/model_figures


# Rename the pdf files to naive, s2s, s2s_reg, stacked
mv test_elec_naive.pdf naive.pdf
mv test_elec_s2s.pdf s2s.pdf
mv test_elec_s2s_reg.pdf s2s_reg.pdf
mv test_elec_stacked.pdf stacked.pdf

# Transform the pdf to png
pdftoppm naive.pdf naive -png
pdftoppm s2s.pdf s2s -png
pdftoppm s2s_reg.pdf s2s_reg -png
pdftoppm stacked.pdf stacked -png