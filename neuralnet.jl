using JLD2, Flux, Statistics
include("loaddata.jl")

# Build model
function create_neuralnet(n::Int, p::Float32)
    return Chain(
        Dense(n => 8, leakyrelu, init=Flux.glorot_uniform),
        Dropout(p),
        Dense(8 => 6, leakyrelu, init=Flux.glorot_uniform),
        Dropout(p),
        Dense(6 => 4, leakyrelu, init=Flux.glorot_uniform),
        Dropout(p),
        Dense(4 => 2, leakyrelu, init=Flux.glorot_uniform),
        Dropout(p),
        Dense(2 => 1, σ, init=Flux.glorot_uniform),
    )
end

# Add & modify data
X, Ystep = loaddata("yelmodata.jld2")
n1, n2, n3 = size(X)
Xperm = permutedims(X, (1, 3, 2))
Yperm = permutedims(Ystep, (1, 3, 2))
n_label = n2*n3
X2d = reshape(Xperm, n1, n_label)[:, 1:n_label]
Y2d = reshape(Yperm, 1, n_label)[:, 1:n_label]

# Norm the data such that all variables x_i have: abs(x_i) <= 1 for all t
normfactor = maximum(abs.(X2d), dims=2)
X2dnorm = X2d ./ normfactor

# Split data into training (70%), validation (15%) and test set (15%)
n_train, n_val = Int(round(n_label*0.7)), Int(round(n_label*0.15))
Xtrain = X2dnorm[:, 1:n_train]
Ytrain = Y2d[:, 1:n_train]
Xval = X2dnorm[:, n_train+1:n_train+n_val]
Yval = Y2d[:, n_train+1:n_train+n_val]
Xtest = X2dnorm[:, n_train+n_val+1:end]
Ytest = Y2d[:, n_train+n_val+1:end]

# Create a data loader to perform batch gradient descent
batchsize = 64
dataloader = Flux.DataLoader((Xtrain, Ytrain), batchsize=batchsize, shuffle=true)

# Define model, optimization criterion and optimizer
model = create_neuralnet(n1, 0.1f0)
loss(m, x::Matrix, y::Matrix) = Flux.Losses.binarycrossentropy(m(x), y)
opt_alg = Flux.Optimise.RAdam(0.0001, (0.9, 0.999))
opt_state = Flux.setup(opt_alg, model)

# Give an idea of model performance
function print_results(when::String)
    println("----------------------------------")
    println("$when the training:")
    println("----------------------------------")
    println("   Loss is $(loss(model, X2dnorm, Y2d))")
    println("   Comparing ̂y and y for some training data points yields:")
    yhat = model(X2dnorm)
    Loss = loss(model, X2dnorm, Y2d)
    for j in vcat(1:10, n_label-10:n_label)
        println("   j = $j: m(x) = $(yhat[:, j][1]),  y = $(Y2d[1, j])")
    end
    println("----------------------------------")
    return yhat, Loss
end

# Show results before the training
yhat_before, loss_before = print_results("Before")

# Define epochs and variables to track loss along the way
n_epochs = 100
epochs = 1:n_epochs
trainloss = fill(Inf, n_epochs)
valloss = fill(Inf, n_epochs)

for epoch in 1:n_epochs
    # Train a whole epoch
    Flux.train!(model, dataloader, opt_state) do m, x, y
        loss(m, x, y)
    end

    # Evaluate the training and validation loss after training over one epoch
    trainloss[epoch] = loss(model, Xtrain, Ytrain)
    valloss[epoch] = loss(model, Xval, Yval)
    println("Training loss at epoch $epoch is $(trainloss[epoch])")
    println("validation loss at epoch $epoch is $(valloss[epoch])")
    
    # Checkpoint the model at each epoch
    jldsave("model_params/checkpoint-$epoch.jld2", model_state = Flux.state(model))
end

# Get lowest validation loss and load the associated parameters from checkpoint files
imin = argmin(valloss)
Flux.loadmodel!(model, JLD2.load("model_params/checkpoint-$imin.jld2", "model_state"))
yhat_after, loss_after = print_results("After")

# Show results after the training
Ytest_hat = model(Xtest)

# Storage all necesary data
struct SplitData{}
    Xtrain::Matrix{Float32}
    Ytrain::Matrix{Float32}
    Xval::Matrix{Float32}
    Yval::Matrix{Float32}
    Xtest::Matrix{Float32}
    Ytest::Matrix{Float32}
    X2dnorm::Matrix{Float32}
    Y2d::Matrix{Float32}
    yhat_before::Matrix{Float32}
    yhat_after::Matrix{Float32}
    Ytest_hat::Matrix{Float32}
    trainloss::Vector{Float32}
    valloss::Vector{Float32}
    n_epochs::Int64
    imin::Int64
    n_label::Int64
    loss_before::Float64
    loss_after::Float64
    epochs::UnitRange{Int64}
end

# Save trained model
using BSON
BSON.@save "waternet.bson" model

# Save data for visualization
data = SplitData(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, X2dnorm, Y2d, yhat_before,
    yhat_after, Ytest_hat, trainloss, valloss, n_epochs, imin, n_label, loss_before,
    loss_after, epochs)
jldsave("data.jld2", data = data)