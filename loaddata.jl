using JLD2, Flux

function loaddata(filename)
    data = JLD2.jldopen("yelmodata.jld2")

    t = data["t"]
    dt = data["dt"]
    vars = data["vars"]
    X = data["X"]
    Ypulse = data["Ypulse"]
    Ystep = data["Ystep"]

    display(vars)
    return X, Ystep, Ypulse
end