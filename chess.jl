using CSV, DataFrames, ChesspicApi, CategoricalArrays, MLDataUtils, Flux, Statistics

# Load the data
games = CSV.read("games.csv", DataFrame)

# Extract the moves for each game
function get_moves(moves_str)
    return split(moves_str, " ")
end
moves_df = DataFrame(
    game_id = games.game_id,
    move = mapreduce(get_moves, vcat, games.moves)
)

# Create a one-hot encoding for the moves
unique_moves = unique(moves_df.move)
moves_onehot = to_categorical(moves_df.move, unique_moves)
moves_onehot_matrix = Matrix{Float64}(undef, length(games.game_id), length(unique_moves))
for (i, game_id) in enumerate(games.game_id)
    game_moves = moves_onehot[moves_df.game_id .== game_id, :]
    moves_onehot_matrix[i, :] = sum(game_moves, dims=1) ./ size(game_moves, 1)
end

# Prepare the input and target variables
X = hcat(Matrix{Float64}(games[:, [:rating_diff, :turns, :white_rating, :black_rating]]), moves_onehot_matrix)
y = games.winner
train_X, test_X, train_y, test_y = splitdata(X, y, Train=0.8, Test=0.2)

# Normalize the input features
train_X_norm = (train_X .- mean(train_X, dims=1)) ./ std(train_X, dims=1)
test_X_norm = (test_X .- mean(train_X, dims=1)) ./ std(train_X, dims=1)

# Define and train the neural network
model = Chain(
    Dense(size(X, 2), 32, relu),
    Dense(32, 16, relu),
    Dense(16, 3)
)
loss(x, y) = Flux.crossentropy(model(x), Flux.onehot(y, 1:3))
opt = Flux.Momentum(0.01)
epochs = 50
for i = 1:epochs
    gs = Flux.gradient(() -> loss(train_X_norm, train_y), Flux.params(model))
    Flux.update!(opt, Flux.params(model), gs)
end

# Evaluate the model
train_accuracy = mean(Flux.onehot(model(train_X_norm), 1:3) .== Flux.onehot(train_y, 1:3))
test_accuracy = mean(Flux.onehot(model(test_X_norm), 1:3) .== Flux.onehot(test_y, 1:3))
println("Train Accuracy: $train_accuracy")
println("Test Accuracy: $test_accuracy")