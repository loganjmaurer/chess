using BayesFlux
using Flux
using CSV
using DataFrames
using Random

# Read the data
data = CSV.read("chess_games.csv", DataFrame)
white_elo = data.white_elo
black_elo = data.black_elo
moves = data.moves
winner = data.Winner

# Encode the moves
move_vocab = unique(vcat(moves...))
move_to_idx = Dict(move => i for (i, move) in enumerate(move_vocab))
encoded_moves = [[move_to_idx[m] for m in game_moves] for game_moves in moves]

# Set the target and input
y = [winner == "White" for winner in winner]
X = encoded_moves

# Randomize the data
perm = randperm(length(X))
X = X[perm]
y = y[perm]

# Split the data into train and test sets
train_size = Int(floor(0.8 * length(X)))
train_X, train_y = X[1:train_size], y[1:train_size]
test_X, test_y = X[train_size+1:end], y[train_size+1:end]

# Set the priors based on Elo ratings
elo_diff = [white_elo[i] - black_elo[i] for i in 1:length(white_elo)]
prior = NetworkPrior(
    Flux.Normal[Flux.Normal(elo_diff[i] / 400, 1) for i in 1:length(move_vocab)],
    Flux.Gamma[Flux.Gamma(1, 1) for _ in 1:64]
)

# Define the model
model = BNN(
    train_X, train_y,
    BNNLikelihood(Flux.Sigmoid()),
    prior,
    BNNInitialiser(Flux.kaiming_normal)
)

# Train the model
train_steps = 1000
for i in 1:train_steps
    Flux.train!(model, [(train_X, train_y)], Flux.ADAM())
    if i % 100 == 0
        println("Iteration $i, Training Loss: $(loss(model, train_X, train_y))")
    end
end

# Evaluate the model on the test set
test_probs = Flux.predict_proba(model, test_X)

# Output the predicted probabilities after each move
predicted_probs = []
for i in 1:length(test_X)
    probs = Flux.predict_proba(model, [test_X[i]])
    push!(predicted_probs, probs[1,1])
end

println("Predicted probabilities after each move:")
println(predicted_probs)

# Calculate the final test accuracy
test_accuracy = mean(Flux.onehot(argmax(test_probs, dims=2), 1:2) .== Flux.onehot(test_y, 1:2))
println("Test Accuracy: $test_accuracy")