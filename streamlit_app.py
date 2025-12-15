"""
Streamlit App - Self-contained version with improved structure

A minimal Streamlit app to run a genetic algorithm
that searches for a sequence of operators.

Getting started:

1) (optional) Create and activate a virtual environment:
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Windows cmd
   .venv\Scripts\activate.bat

2) Install dependencies:
   pip install streamlit numpy matplotlib

3) Execute the app:
   streamlit run streamlit_app.py

4) Navigate to the app's URL (default http://localhost:8501).
"""

import streamlit as st
import random
import numpy as np
from typing import List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ==================== DATA CLASSES ====================

@dataclass
class GeneticConfig:
    """Configuration for the genetic algorithm."""
    population_size: int
    max_generations: int
    tournament_size: int
    replacement_ratio: float
    crossover_probability: float
    mutation_probability: float
    
    def validate(self):
        """Validate configuration parameters."""
        if self.population_size < 2:
            raise ValueError("Population size must be at least 2")
        if self.max_generations < 1:
            raise ValueError("Max generations must be at least 1")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("Crossover probability must be between 0 and 1")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("Mutation probability must be between 0 and 1")
        if not 0 < self.replacement_ratio <= 1:
            raise ValueError("Replacement ratio must be between 0 and 1")


@dataclass
class Problem:
    """Definition of the optimization problem."""
    numbers: List[int]
    target: int
    operators: List[str]
    
    def validate(self):
        """Validate problem parameters."""
        if len(self.numbers) < 2:
            raise ValueError("At least 2 numbers are required")
        if len(self.operators) < 1:
            raise ValueError("At least 1 operator is required")
    
    @property
    def chromosome_length(self) -> int:
        """Length of the chromosome (number of operators needed)."""
        return len(self.numbers) - 1


# ==================== INDIVIDUAL CLASS ====================

class Individual:
    """Represents an individual in the population."""
    
    # Operator mapping - supports extensible operators
    OPERATOR_FUNCTIONS = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '×': lambda x, y: x * y,
        '/': lambda x, y: x / y,
        '÷': lambda x, y: x / y,
        '^': lambda x, y: x ** y,
        '**': lambda x, y: x ** y,
        '%': lambda x, y: x % y,
    }
    
    def __init__(self, genes: List[int], problem: Problem):
        """
        Initialize an individual.
        
        Args:
            genes: List of operator indices
            problem: Problem definition
        """
        self.genes = genes
        self.problem = problem
        self._fitness = None
        self._result = None
    
    @classmethod
    def random(cls, problem: Problem) -> 'Individual':
        """Create a random individual."""
        genes = [random.randint(0, len(problem.operators) - 1) 
                for _ in range(problem.chromosome_length)]
        return cls(genes, problem)
    
    def evaluate(self) -> float:
        """
        Evaluate the individual's fitness (cached).
        
        Returns:
            Fitness score (lower is better)
        """
        if self._fitness is not None:
            return self._fitness
        
        # Maximum safe value to prevent overflow
        MAX_SAFE_VALUE = 1e15
        PENALTY_FITNESS = 1e9
        
        try:
            result = self.problem.numbers[0]
            
            for i, gene in enumerate(self.genes):
                operator_symbol = self.problem.operators[gene]
                operator_func = self.OPERATOR_FUNCTIONS.get(operator_symbol)
                
                if operator_func is None:
                    raise ValueError(f"Unknown operator: {operator_symbol}")
                
                result = operator_func(result, self.problem.numbers[i + 1])
                
                # Check for invalid or extreme results
                if isinstance(result, complex):
                    self._fitness = PENALTY_FITNESS
                    self._result = None
                    return self._fitness
                
                if not np.isfinite(result) or abs(result) > MAX_SAFE_VALUE:
                    self._fitness = PENALTY_FITNESS
                    self._result = None
                    return self._fitness
            
            self._result = result
            
            # Calculate fitness with overflow protection
            if abs(self.problem.target - result) > MAX_SAFE_VALUE:
                self._fitness = PENALTY_FITNESS
            else:
                self._fitness = abs(self.problem.target - result)
            
        except (ZeroDivisionError, OverflowError, ValueError, TypeError):
            self._fitness = PENALTY_FITNESS
            self._result = None
        
        return self._fitness
    
    @property
    def fitness(self) -> float:
        """Get fitness (evaluates if needed)."""
        if self._fitness is None:
            return self.evaluate()
        return self._fitness
    
    @property
    def result(self) -> Optional[float]:
        """Get the calculated result."""
        if self._result is None and self._fitness is None:
            self.evaluate()
        return self._result
    
    def mutate(self, probability: float) -> 'Individual':
        """
        Mutate the individual's genes.
        
        Args:
            probability: Probability of mutating each gene
            
        Returns:
            Self (for method chaining)
        """
        for i in range(len(self.genes)):
            if random.random() < probability:
                self.genes[i] = random.randint(0, len(self.problem.operators) - 1)
        
        # Invalidate cached values
        self._fitness = None
        self._result = None
        return self
    
    def copy(self) -> 'Individual':
        """Create a deep copy of this individual."""
        return Individual(self.genes.copy(), self.problem)
    
    def to_expression(self) -> str:
        """
        Convert individual to mathematical expression string.
        
        Returns:
            String representation like "((((25 + 6) × 9) + 75) × 50) - 3"
        """
        if len(self.genes) == 0:
            return str(self.problem.numbers[0])
        
        # Build expression with proper parentheses
        expr = str(self.problem.numbers[0])
        
        for i, gene in enumerate(self.genes):
            operator = self.problem.operators[gene]
            number = self.problem.numbers[i + 1]
            expr = f"({expr} {operator} {number})"
        
        return expr
    
    def __str__(self) -> str:
        """String representation of the individual."""
        expr = self.to_expression()
        result = self.result if self.result is not None else "Error"
        
        if isinstance(result, float):
            # Format with commas for thousands
            if result == int(result):
                result_str = f"{int(result):,}"
            else:
                result_str = f"{result:,.2f}"
        else:
            result_str = str(result)
        
        return f"{expr} = {result_str}"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"Individual(genes={self.genes}, fitness={self.fitness:.2f})"


# ==================== POPULATION CLASS ====================

class Population:
    """Manages a population of individuals."""
    
    def __init__(self, individuals: List[Individual]):
        """Initialize population with individuals."""
        self.individuals = individuals
        self._sorted = False
    
    @classmethod
    def random(cls, size: int, problem: Problem) -> 'Population':
        """Create a random population."""
        individuals = [Individual.random(problem) for _ in range(size)]
        return cls(individuals)
    
    def evaluate(self) -> List[float]:
        """Evaluate all individuals and return fitness scores."""
        return [ind.fitness for ind in self.individuals]
    
    def sort_by_fitness(self):
        """Sort population by fitness (best first)."""
        if not self._sorted:
            self.individuals.sort(key=lambda ind: ind.fitness)
            self._sorted = True
    
    @property
    def best(self) -> Individual:
        """Get the best individual."""
        self.sort_by_fitness()
        return self.individuals[0]
    
    @property
    def worst(self) -> Individual:
        """Get the worst individual."""
        self.sort_by_fitness()
        return self.individuals[-1]
    
    @property
    def size(self) -> int:
        """Get population size."""
        return len(self.individuals)
    
    def tournament_selection(self, tournament_size: int, count: int) -> List[Individual]:
        """
        Select individuals using tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        selected = []
        available_indices = list(range(self.size))
        
        for _ in range(count):
            if not available_indices:
                break
            
            # Select random individuals for tournament
            tournament_indices = random.sample(
                available_indices, 
                min(tournament_size, len(available_indices))
            )
            
            # Find best in tournament
            tournament = [self.individuals[i] for i in tournament_indices]
            winner = min(tournament, key=lambda ind: ind.fitness)
            
            # Remove winner from available pool (no replacement within generation)
            winner_idx = self.individuals.index(winner)
            if winner_idx in available_indices:
                available_indices.remove(winner_idx)
            
            selected.append(winner.copy())
        
        return selected
    
    def replace_worst(self, new_individuals: List[Individual]):
        """Replace worst individuals with new ones."""
        self.sort_by_fitness()
        num_to_replace = len(new_individuals)
        self.individuals = self.individuals[:-num_to_replace] + new_individuals
        self._sorted = False


# ==================== GENETIC OPERATORS ====================

class GeneticOperators:
    """Genetic algorithm operators."""
    
    @staticmethod
    def crossover_single_point(parent1: Individual, parent2: Individual, 
                               probability: float) -> Tuple[Individual, Individual]:
        """
        Single-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            probability: Probability of performing crossover
            
        Returns:
            Two offspring
        """
        if random.random() < probability and len(parent1.genes) > 1:
            point = random.randint(1, len(parent1.genes) - 1)
            
            child1_genes = parent1.genes[:point] + parent2.genes[point:]
            child2_genes = parent2.genes[:point] + parent1.genes[point:]
            
            child1 = Individual(child1_genes, parent1.problem)
            child2 = Individual(child2_genes, parent2.problem)
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        
        return child1, child2
    
    @staticmethod
    def crossover_uniform(parent1: Individual, parent2: Individual, 
                         probability: float) -> Tuple[Individual, Individual]:
        """
        Uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            probability: Probability of performing crossover
            
        Returns:
            Two offspring
        """
        if random.random() < probability:
            mask = [random.randint(0, 1) for _ in range(len(parent1.genes))]
            
            child1_genes = [p1 if m == 0 else p2 
                           for p1, p2, m in zip(parent1.genes, parent2.genes, mask)]
            child2_genes = [p2 if m == 0 else p1 
                           for p1, p2, m in zip(parent1.genes, parent2.genes, mask)]
            
            child1 = Individual(child1_genes, parent1.problem)
            child2 = Individual(child2_genes, parent2.problem)
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        
        return child1, child2


# ==================== GENETIC ALGORITHM ====================

class GeneticAlgorithm:
    """Main genetic algorithm implementation."""
    
    def __init__(self, problem: Problem, config: GeneticConfig):
        """
        Initialize genetic algorithm.
        
        Args:
            problem: Problem definition
            config: Algorithm configuration
        """
        self.problem = problem
        self.config = config
        self.population = None
        self.generation = 0
        self.best_individual = None
        self.best_fitness_history = []
        self.average_fitness_history = []
        
        # Validate inputs
        self.problem.validate()
        self.config.validate()
    
    def initialize(self):
        """Initialize the population."""
        self.population = Population.random(self.config.population_size, self.problem)
        self.generation = 0
        self.best_fitness_history = []
        self.average_fitness_history = []
    
    def evolve_generation(self):
        """Evolve one generation."""
        # Evaluate population
        fitness_scores = self.population.evaluate()
        
        # Track statistics (convert to float64 to handle large numbers safely)
        fitness_array = np.array(fitness_scores, dtype=np.float64)
        best_fitness = float(np.min(fitness_array))
        avg_fitness = float(np.mean(fitness_array))
        
        self.best_fitness_history.append(best_fitness)
        self.average_fitness_history.append(avg_fitness)
        
        # Update best individual
        if self.best_individual is None or best_fitness < self.best_individual.fitness:
            self.best_individual = self.population.best.copy()
        
        # Check termination
        if best_fitness == 0:
            return True  # Perfect solution found
        
        # Selection
        num_offspring = int(self.config.population_size * self.config.replacement_ratio)
        num_offspring = max(2, num_offspring - (num_offspring % 2))  # Ensure even number
        
        parents = self.population.tournament_selection(
            self.config.tournament_size, 
            num_offspring
        )
        
        # Crossover and mutation
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = GeneticOperators.crossover_single_point(
                    parents[i], 
                    parents[i + 1], 
                    self.config.crossover_probability
                )
                
                child1.mutate(self.config.mutation_probability)
                child2.mutate(self.config.mutation_probability)
                
                offspring.extend([child1, child2])
        
        # Replacement
        self.population.replace_worst(offspring)
        self.generation += 1
        
        return False  # Continue evolution
    
    def run(self, progress_callback: Optional[Callable] = None) -> Individual:
        """
        Run the genetic algorithm.
        
        Args:
            progress_callback: Optional callback for progress updates
                             Signature: callback(generation, max_generations, best_fitness)
        
        Returns:
            Best individual found
        """
        self.initialize()
        
        for gen in range(self.config.max_generations):
            terminated = self.evolve_generation()
            
            if progress_callback:
                progress_callback(
                    gen + 1, 
                    self.config.max_generations, 
                    self.best_individual.fitness
                )
            
            if terminated:
                break
        
        return self.best_individual


# ==================== STREAMLIT APP ====================

st.set_page_config(
    page_title="Genetic Search Algorithms",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Search Algorithms")
st.markdown("---")

st.subheader("What is a Genetic Algorithm?")
st.markdown("""
A **genetic algorithm** (*from now on GA*) is a search heuristic inspired by the process of natural selection.
It is used to find approximate solutions to optimization and search problems.
GAs simulate the process of evolution by using techniques such as selection, crossover, and mutation to evolve a population of candidate solutions over generations.
""")

with st.expander("How does it work?"):
    st.image("assets\ag_flux_diagram.svg", caption="Genetic Algorithm Flow Diagram")
    st.markdown("""
    1. **Initialization**: A population of candidate solutions (individuals) is randomly generated.
    2. **Evaluation**: Each individual is evaluated using a **fitness function** that measures how well it solves the problem.
    3. **Genetic Operators**: Each iteration creates a new generation:
        - **Selection**: Individuals are selected based on their fitness scores to create a mating pool
        - **Crossover**: Pairs of individuals from the mating pool are combined to produce offspring, inheriting traits from both parents.
        - **Mutation**: Random changes are introduced to some individuals to maintain genetic diversity.
        - **Replacement**: The new generation of individuals replaces the old population.
    4. **Termination**: The process repeats until a stopping criteria is met (Normally, achieving a certain fitness score or a maximum number of generations).
    """)

st.subheader("The problem to solve:")

st.info(
    """
    **Operator Sequence Search**

    The problem asks you to take a sequence of given integers and use a set of arithmetic operators to get a result that is as close as possible to a target integer.

    **Example:** 
    - Given Integers: 25, 6, 9, 75, 50, 3 
    - Target: 307
    

    > **Note:** We assume **left-to-right sequential calculation** for simplicity.
    > A sequence of operators like: +, ×, +, ×, - is interpreted as: ((((25 + 6) × 9) + 75) × 50) - 3 = 14,352

    """,
    icon="🔢",
)

st.markdown("---")

st.subheader("Genetic Search Algorithm")

with st.expander("Problem Representation:"):
    st.markdown("""
    **Individuals**: An individual is a sequence of arithmetic operators (e.g., +, ×, +, ×, -).
                
    **Fitness Function**: Fitness is based on minimizing the difference between the expression's result and the target value.
    
    **Genetic Operators**:
    - Selection: Tournament Selection is used. A group of $k$ individuals is randomly chosen, and the best one is selected for reproduction.
    - Crossover: A crossover operation combines two operator sequences with a configurable probability.
    - Mutation: A mutation operation randomly changes an operator in the sequence with a configurable probability.
    """)

# Sidebar parameters
with st.sidebar:
    st.header("Problem Parameters")
    lista_text = st.text_area("Given Integers: (comma-separated)", value="75,3,1,4,50,6,12,8", height=80)
    objetivo = st.number_input("Target", value=852)
    operators_text = st.text_input("Operators (comma-separated)", value="+,-,*,/")
    st.caption(f"Supported operators: {', '.join(Individual.OPERATOR_FUNCTIONS.keys())}")
    run_button = st.button("Run GA", type="primary")
    
    st.markdown("---")
    st.header("GA Parameters")
    n_inicial = st.number_input("Initial population size", value=100, min_value=2)
    n_epocas = st.number_input("Max generations", value=100, min_value=1)
    prob_cruce = st.slider("Crossover Probability", 0.0, 1.0, 0.6)
    prob_mutacion = st.slider("Mutation Probability", 0.0, 1.0, 0.05)
    replacement_ratio = st.slider("Replacement Ratio", 0.1, 1.0, 0.5)
    k_torneo = st.number_input("Tournament size", value=3, min_value=1)
    st.markdown("---")
    st.caption("💡 Tip: Try different parameter combinations to see how they affect convergence!")

# Main execution
if run_button:
    # Parse input
    try:
        numbers = [int(x.strip()) for x in lista_text.split(",") if x.strip()]
        operators = [op.strip() for op in operators_text.split(",") if op.strip()]
        
        if len(numbers) < 2:
            st.error("You need at least 2 numbers in the list.")
            st.stop()
        
        if len(operators) < 1:
            st.error("You need at least 1 operator.")
            st.stop()
        
        # Validate operators
        for op in operators:
            if op not in Individual.OPERATOR_FUNCTIONS:
                st.error(f"Unknown operator: '{op}'. Supported operators: {', '.join(Individual.OPERATOR_FUNCTIONS.keys())}")
                st.stop()
        
    except ValueError as e:
        st.error(f"Invalid input format: {str(e)}")
        st.stop()

    # Create problem and configuration
    problem = Problem(numbers=numbers, target=int(objetivo), operators=operators)
    config = GeneticConfig(
        population_size=int(n_inicial),
        max_generations=int(n_epocas),
        tournament_size=int(k_torneo),
        replacement_ratio=float(replacement_ratio),
        crossover_probability=float(prob_cruce),
        mutation_probability=float(prob_mutacion)
    )
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(generation, max_generations, best_fitness):
        progress_bar.progress(generation / max_generations)
        status_text.text(f"Generation {generation}/{max_generations} - Best fitness: {best_fitness:.2f}")
    
    # Run genetic algorithm
    try:
        with st.spinner("Running genetic algorithm..."):
            random.seed()  # Ensure randomness between runs
            
            ga = GeneticAlgorithm(problem, config)
            best_individual = ga.run(progress_callback=update_progress)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        if best_individual is None:
            st.warning("The execution did not return a valid solution.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"✅ Best fitness: {best_individual.fitness:.2f}")
                st.markdown(f"**Best solution:**")
                st.code(str(best_individual), language=None)
                
                if best_individual.fitness == 0:
                    st.success("🎉 Perfect solution found!")
                else :
                    st.caption("""
                               No optimal solution was found. 
                               The problem may be too complex for the current algorithm, 
                               or a perfect solution may not exist.

                               If you believe an optimal solution exists,
                               consider increasing the population size or number of generations,
                               or experimenting with different operator configurations.""")
            
            with col2:
                st.metric("Generations executed", ga.generation)
                st.metric("Final average fitness", f"{ga.average_fitness_history[-1]:.2f}")
                st.metric("Population size", ga.population.size)
            
            # Plot convergence
            st.markdown("---")
            st.subheader("Convergence Analysis")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            generations = range(len(ga.best_fitness_history))
            
            # Best fitness plot
            ax1.plot(generations, ga.best_fitness_history, color='red', linewidth=2, label='Best')
            ax1.set_title('Best Fitness per Generation')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness (Error)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Average fitness plot
            ax2.plot(generations, ga.average_fitness_history, color='blue', linewidth=2, label='Average')
            ax2.set_title('Average Fitness per Generation')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Fitness (Error)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show top 5 individuals
            with st.expander("View Top 5 Solutions"):
                ga.population.sort_by_fitness()
                for i, individual in enumerate(ga.population.individuals[:5], 1):
                    st.text(f"{i}. {individual} (fitness: {individual.fitness:.2f})")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")

st.caption("""
If you want to learn more about: 
- The impact of each parameter, and how to tune them
- Other selection, crossover, and mutation strategies
- The implementation details
- Additional experiments and results, Including a 1000 numbers test case

I strongly recommend checking the original Jupyter Notebook:
https://github.com/RubenGonV/Simulacion/blob/main/P2/genetic_main.ipynb
(some comments are in Spanish, but the code is universal 😉)
""")