using BSON, JLD2, CairoMakie, Statistics, Flux

## Cargamos los datos
Core.eval(Main, :(import NNlib))
BSON.@load "waternet.bson" model
data = JLD2.load("data.jld2", "data")

# Nombramos los datos
Xtrain = data.Xtrain; Ytrain = data.Ytrain;
Xval = data.Xval; Yval = data.Yval
Xtest = data.Xtest; Ytest = data.Ytest;
X2dnorm = data.X2dnorm; Y2d = data.Y2d;
yhat_before = data.yhat_before; yhat_after = data.yhat_after;
Ytest_hat = data.Ytest_hat;
trainloss = data.trainloss; valloss = data.valloss;
n_epochs = data.n_epochs;
imin = data.imin; n_label = data.n_label;
loss_before = data.loss_before; loss_after = data.loss_after;
epochs = data.epochs; 

##########################################################################################

fig = Figure()
ax = Axis(fig[1, 1], limits = ((0.7, 10.3), (-0.1, 1.1)))
scatter!(ax, 1:10, vec(Y2d)[1:10], label = "ground truth", markersize = 10, color = :black)
scatter!(ax, 1:10, vec(yhat_before)[1:10], label = "ŷ before training", markersize = 10, color = :red)
text!(1.5, 0.75, text = "loss before training is $(round.(loss_before; digits=3))", align = (:left, :center))
ax.xlabel = "time (x10 years, first 10 values of first experiment)"; ax.xlabelsize = 20
ax.ylabel = "y (output)"; ax.ylabelsize = 20
axislegend(ax, position = (0.9, 0.75))
ax.subtitle = "Comparación datos de terreno y predicción previa al entrenamiento"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

fig = Figure()
ax = Axis(fig[1, 1], limits = (nothing, (-0.1, 1.1)))
scatter!(ax, n_label-10:n_label, vec(Y2d)[end-10:end], label = "ground truth", markersize = 10, color = :black)
scatter!(ax, n_label-10:n_label, vec(yhat_before)[end-10:end], label = "ŷ before training", markersize = 10, color = :red)
text!(n_label-9.5, 0.75, text = "loss before training is $(round.(loss_before; digits=3))", align = (:left, :center))
ax.xlabel = "time (x10 years, last 10 values of last experiment)"; ax.xlabelsize = 20
ax.ylabel = "y (output)"; ax.ylabelsize = 20
axislegend(ax, position = (0.9, 0.75))
ax.subtitle = "Comparación datos de terreno y predicción previa al entrenamiento"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

fig = Figure()
ax = Axis(fig[1, 1], limits = (nothing, (-0.1, 1.1)))
scatter!(ax, 1:10, vec(Y2d)[1:10], label = "ground truth", markersize = 15, color = :black)
scatter!(ax, 1:10, vec(yhat_after)[1:10], label = "ŷ after training", markersize = 10, color = :orange)
text!(1.5, 0.75, text = "loss after training is $(round.(loss_after; digits=3))", align = (:left, :center))
ax.xlabel = "time (x10 years, first 10 values of first experiment)"; ax.xlabelsize = 20
ax.ylabel = "y (output)"; ax.ylabelsize = 20
axislegend(ax, position = (0.9, 0.75))
ax.subtitle = "Comparación datos de terreno y predicción posterior al entrenamiento"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

fig = Figure()
ax = Axis(fig[1, 1], limits = (nothing, (-0.1, 1.1)))
scatter!(ax, n_label-10:n_label, vec(Y2d)[end-10:end], label = "ground truth", markersize = 15, color = :black)
scatter!(ax, n_label-10:n_label, vec(yhat_after)[end-10:end], label = "ŷ after training", markersize = 10, color = :orange)
text!(n_label-9.5, 0.75, text = "loss after training is $(round.(loss_after; digits=3))", align = (:left, :center))
ax.xlabel = "time (x10 years, last 10 values of last experiment)"; ax.xlabelsize = 20
ax.ylabel = "y (output)"; ax.ylabelsize = 20
axislegend(ax, position = (0.9, 0.75))
ax.subtitle = "Comparación datos de terreno y predicción posterior al entrenamiento"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

##########################################################################################

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, epochs, trainloss, label = "train", linewidth = 3, color = :blue)
lines!(ax, epochs, valloss, label = "validation", linewidth = 3, color = :orange)
vlines!(ax, imin, label = "epoch nº$(imin)", linestyle = :dash, color = :grey)
ax.xlabel = "epochs"; ax.xlabelsize = 20
ax.ylabel = "loss"; ax.ylabelsize = 20
axislegend(ax, position = (0.9, 0.85))
ax.subtitle = "''loss'' para los conjuntos de entrenamiento y validación"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

# Here we could optionally set a logarithmic y axis for nicer visualization.
fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
lines!(ax, epochs, trainloss, label = "train", linewidth = 3, color = :blue)
lines!(ax, epochs, valloss, label = "validation", linewidth = 3, color = :orange)
vlines!(ax, imin, label = "epoch nº$(imin)", linestyle = :dash, color = :grey)
ax.xlabel = "epochs"; ax.xlabelsize = 20
ax.ylabel = "loss (logarithmic scale)"; ax.ylabelsize = 20
axislegend(ax, position = (0.9, 0.85))
ax.subtitle = "''loss'' para conjuntos de entrenamiento y validación, escala logarítmica"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

##########################################################################################
# Visualization for the test set!

fig = Figure()   # PRUEBA 1
ax = Axis(fig[1, 1])
lines!(ax,  1:3001, vec(Ytest_hat)[1:3001], label = "model prediction", linewidth = 3, color = :green)
lines!(ax, 1:3001, vec(Ytest)[1:3001], label = "ground truth", linewidth = 3, color = :brown)
text!(2500, 0.63, text = "PRUEBA 1", align = (:center, :center))
ax.xlabel = "time (x10 years)"; ax.xlabelsize = 20
ax.ylabel = "y (output)"; ax.ylabelsize = 20
axislegend(ax, position = (0.91, 0.5))
ax.subtitle = "Comparación datos de terreno y predicción del modelo"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

fig = Figure()   # ZOOM PRUEBA 1
ax = Axis(fig[1, 1])
lines!(ax,  650:1000, vec(Ytest_hat)[650:1000], label = "model prediction", linewidth = 3, color = :green)
lines!(ax, 650:1000, vec(Ytest)[650:1000], label = "ground truth", linewidth = 3, color = :brown)
text!(950, 0.63, text = "ZOOM PRUEBA 1", align = (:center, :center))
ax.xlabel = "time (x10 years)"; ax.xlabelsize = 20
ax.ylabel = "y (output)"; ax.ylabelsize = 20
axislegend(ax, position = (0.94, 0.5))
ax.subtitle = "Comparación datos de terreno y predicción del modelo"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

fig = Figure()   # PRUEBA 2
ax = Axis(fig[1, 1])
lines!(ax,  1:3001, vec(Ytest_hat)[3002:6002], label = "model prediction", linewidth = 3, color = :green)
lines!(ax, 1:3001, vec(Ytest)[3002:6002], label = "ground truth", linewidth = 3, color = :brown)
text!(2500, 0.63, text = "PRUEBA 2", align = (:center, :center))
ax.xlabel = "time (x10 years)"; ax.xlabelsize = 20
ax.ylabel = "y (output)"; ax.ylabelsize = 20
axislegend(ax, position = (0.91, 0.5))
ax.subtitle = "Comparación datos de terreno y predicción del modelo"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

fig = Figure()   # ZOOM PRUEBA 2
ax = Axis(fig[1, 1])
lines!(ax,  700:1100, vec(Ytest_hat)[3700:4100], label = "model prediction", linewidth = 3, color = :green)
lines!(ax, 700:1100, vec(Ytest)[3700:4100], label = "ground truth", linewidth = 3, color = :brown)
text!(1050, 0.63, text = "ZOOM PRUEBA 2", align = (:center, :center))
ax.xlabel = "time (x10 years)"; ax.xlabelsize = 20
ax.ylabel = "y (output)"; ax.ylabelsize = 20
axislegend(ax, position = (0.96, 0.5))
ax.subtitle = "Comparación datos de terreno y predicción del modelo"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

fig = Figure()   # PRUEBA 3
ax = Axis(fig[1, 1], limits = (nothing, (-0.1, 1.1)))
lines!(ax,  1:3001, vec(Ytest_hat)[9004:12004], label = "model prediction", linewidth = 3, color = :green)
lines!(ax, 1:3001, vec(Ytest)[9004:12004], label = "ground truth", linewidth = 3, color = :brown)
text!(2500, 0.63, text = "PRUEBA 3", align = (:center, :center))
ax.xlabel = "time (x10 years)"; ax.xlabelsize = 20
ax.ylabel = "y (output)"; ax.ylabelsize = 20
axislegend(ax, position = (0.91, 0.5))
ax.subtitle = "Comparación datos de terreno y predicción del modelo"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig

fig = Figure()   # PRUEBA 4
ax = Axis(fig[1, 1], limits = (nothing, (-0.1, 1.1)))
lines!(ax,  1:3001, vec(Ytest_hat)[12005:15005], label = "model prediction", linewidth = 3, color = :green)
lines!(ax, 1:3001, vec(Ytest)[12005:15005], label = "ground truth", linewidth = 3, color = :brown)
text!(2500, 0.63, text = "PRUEBA 4", align = (:center, :center))
ax.xlabel = "time (x10 years)"; ax.xlabelsize = 20
ax.ylabel = "y (output)"; ax.ylabelsize = 20
axislegend(ax, position = (0.91, 0.5))
ax.subtitle = "Comparación datos de terreno y predicción del modelo"; ax.subtitlesize = 20
ax.xticklabelsize = 20; ax.yticklabelsize = 20
fig