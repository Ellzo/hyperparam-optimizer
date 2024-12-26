from random import sample
import torch
import numpy as np

class GeneticTraining:
    def __init__(self, train_data, val_data, model_class, device=None):
        self.train_data = train_data
        self.val_data = val_data
        self.model_class = model_class
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fitness_map = {}

    def train(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def evaluate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return val_loss / len(val_loader), 100.0 * correct / len(val_loader.dataset)

    def fitness_function(self, individual, generation, criterion=torch.nn.CrossEntropyLoss(), verbose=0, metric='accuracy'):
        enc_individual = "_".join(f"{key}={val}" for key, val in individual.items())

        if enc_individual in self.fitness_map:
            return self.fitness_map[enc_individual]["val_accuracy"]

        model = self.model_class().to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=individual["lr"])
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=individual["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=individual["batch_size"], shuffle=False)

        if verbose > 0:
            print(f"[Generation {generation}] Training with {individual}")

        for epoch in range(individual["n_epochs"]):
            self.train(model, train_loader, criterion, optimizer)

        val_loss, accuracy = self.evaluate(model, val_loader, criterion)

        self.fitness_map[enc_individual] = {
            "generation": generation,
            **individual,
            "val_loss": val_loss,
            "val_accuracy": accuracy,
        }

        if verbose > 0:
            print(f"[Generation {generation}] Validation Accuracy: {accuracy:.2f}%")

        if metric == 'accuracy':
            return accuracy
        else:
            return -val_loss

    def generate_population(self, hyperparams, population_size):
        population = []
        for _ in range(population_size):
            individual = {
                key: torch.rand(1).item() * (param["max"] - param["min"]) + param["min"]
                if param["type"] == "float"
                else torch.randint(int(param["min"]), int(param["max"] + 1), (1,)).item()
                for key, param in hyperparams.items()
            }
            population.append(individual)
        return population

    def mutate(self, individual, hyperparams, mutation_std_factor):
        mutated = {}
        for key, value in individual.items():
            if hyperparams[key]["type"] == "float":
                mutated[key] = value + np.random.normal(0, (hyperparams[key]["max"] - hyperparams[key]["min"]) / mutation_std_factor)
                mutated[key] = np.clip(mutated[key], hyperparams[key]["min"], hyperparams[key]["max"])
            elif hyperparams[key]["type"] == "int":
                mutated[key] = int(value + np.random.normal(0, (hyperparams[key]["max"] - hyperparams[key]["min"]) / mutation_std_factor))
                mutated[key] = max(hyperparams[key]["min"], min(mutated[key], hyperparams[key]["max"]))
            elif hyperparams[key]["type"] == "exp":
                mutated[key] = int(value + np.random.normal(0, 10**(-np.log10(hyperparams[key]["max"]) + np.log10(hyperparams[key]["min"]) - 1)))
                mutated[key] = max(hyperparams[key]["min"], min(mutated[key], hyperparams[key]["max"]))
        return mutated

    def crossover(self, parent1, parent2):
        return {
            key: (parent1[key] + parent2[key]) // 2
            if hyperparams[key]["type"] == "int"
            else (parent1[key] + parent2[key]) / 2
            for key in parent1.keys() }

    def run_evolution(self, population_size, num_generations, hyperparams, crossover_rate, mutation_rate, mutation_std_factor, verbose=0):
        population = self.generate_population(hyperparams, population_size)

        for generation in range(num_generations):
            if verbose > 0:
                print(f"\nStarting generation {generation + 1}...")

            fitness_scores = [
                self.fitness_function(individual, generation + 1, verbose=verbose)
                for individual in population
            ]

            sorted_indices = np.argsort(fitness_scores)[::-1]
            selected_population = [population[i] for i in sorted_indices[:int(crossover_rate * population_size)]]

            offspring = []
            while len(offspring) < (population_size - len(selected_population)):
                p1, p2 = sample(selected_population, 2)
                child = self.crossover(p1, p2)
                if torch.rand(1).item() < mutation_rate:
                    child = self.mutate(child, hyperparams, mutation_std_factor)
                offspring.append(child)

            population = selected_population + offspring

        best_individual = max(population, key=lambda ind: self.fitness_function(ind, num_generations))
        if verbose > 0:
            print("\nEvolution complete.")
            print(f"Best individual: {best_individual}")
        return best_individual