# src/genetic_optimization.py
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms

# Load the extracted features
print("Loading features...")    
with open('data/features/audio_features.pkl', 'rb') as f:
    X, y = pickle.load(f)

# Convert emotions to numerical labels
emotions = np.unique(y)
emotion_to_label = {emotion: i for i, emotion in enumerate(emotions)}
y_encoded = np.array([emotion_to_label[emotion] for emotion in y])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the genetic algorithm components
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Feature selection: each gene represents whether to include a feature
num_features = X.shape[1]
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluate function for genetic algorithm
def evaluate(individual):
    # Get selected features
    selected_indices = [i for i, include in enumerate(individual) if include]
    
    # If no features selected, return low fitness
    if len(selected_indices) == 0:
        return 0.0,
    
    # Select subset of features
    X_subset = X_scaled[:, selected_indices]
    
    # Use cross-validation to evaluate model with selected features
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    scores = cross_val_score(clf, X_subset, y_encoded, cv=5, scoring='accuracy')
    
    return np.mean(scores),

# Register genetic operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm parameters
population_size = 50
num_generations = 20
crossover_prob = 0.7
mutation_prob = 0.2

# Run the genetic algorithm
print("\n--- Running Genetic Algorithm for Feature Selection ---")
population = toolbox.population(n=population_size)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# Run evolutionary algorithm
pop, log = algorithms.eaSimple(population, toolbox, 
                              cxpb=crossover_prob, 
                              mutpb=mutation_prob, 
                              ngen=num_generations, 
                              stats=stats, 
                              halloffame=hof, 
                              verbose=True)

# Get best solution
best_individual = hof[0]
selected_features = [i for i, include in enumerate(best_individual) if include]
num_selected = len(selected_features)

print(f"\nGenetic algorithm complete!")
print(f"Selected {num_selected} out of {num_features} features")
print(f"Best accuracy: {best_individual.fitness.values[0]:.4f}")

# Train model with optimized features
X_optimized = X_scaled[:, selected_features]
model = RandomForestClassifier(n_estimators=100)
model.fit(X_optimized, y_encoded)

# Save optimized model and selected features
with open('models/ga_optimized_model.pkl', 'wb') as f:
    pickle.dump((model, scaler, selected_features, emotions), f)

# Plot feature importance for selected features
plt.figure(figsize=(10, 6))
feature_importances = model.feature_importances_
sorted_idx = np.argsort(feature_importances)
plt.barh(range(num_selected), feature_importances[sorted_idx])
plt.yticks(range(num_selected), [f"Feature {selected_features[i]}" for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance (GA-Selected Features)')
plt.tight_layout()
plt.savefig('models/ga_feature_importance.png')

# Plot evolution statistics
gen = log.select("gen")
fit_mins = log.select("min")
fit_avgs = log.select("avg")
fit_maxs = log.select("max")

plt.figure(figsize=(10, 6))
plt.plot(gen, fit_mins, label="Min Fitness")
plt.plot(gen, fit_avgs, label="Avg Fitness")
plt.plot(gen, fit_maxs, label="Max Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness (Accuracy)")
plt.legend(loc="lower right")
plt.title("Genetic Algorithm Convergence")
plt.tight_layout()
plt.savefig('models/ga_convergence.png')

print("\nOptimized model saved successfully!")