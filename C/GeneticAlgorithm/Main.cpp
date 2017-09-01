/*********************************************************************

This code is basically the one made by the by John LeFlohic but updated with the amazing STL Library and some minor adjustments to show results
while running de generations;

Link: http://www-cs-students.stanford.edu/~jl/Essays/ga.html

Purpose: Simple Genetic Algorith in C++

@author Angelo Antonio Manzatto
@version 1.0 01/09/2017

*********************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>
#include <string>

#define NUMBER_ORGANISMS 100
#define NUMBER_GENES 20
#define ALLELES 4
#define MUTATION_RATE 0.001
#define MAXIMUM_FITNESS NUMBER_GENES

std::vector<std::vector<char>> m_currentGeneration;
std::vector<std::vector<char>> m_nextGeneration;
std::vector<char> m_modelOrganism;
std::vector<int> m_organismsFitness;

int m_totalOfFitnesses;

char genome[4] = { 'A','G','C','T' };

inline char GetRandomGenome()
{

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> dis(0,sizeof(genome)-1);

	return genome[dis(gen)];
}


void AllocateMemory()
{
	m_currentGeneration.resize(NUMBER_ORGANISMS);
	m_nextGeneration.resize(NUMBER_ORGANISMS);
	m_modelOrganism.resize(NUMBER_GENES);
	m_organismsFitness.resize(NUMBER_ORGANISMS);

	for (int organism = 0; organism < NUMBER_ORGANISMS; organism++)
	{
		m_currentGeneration[organism].resize(NUMBER_GENES);
		m_nextGeneration[organism].resize(NUMBER_GENES);
	}
}

void InitializeOrganisms()
{
	int organisms;
	int genes;

	// Initialize the normal organisms
	for (organisms = 0; organisms < NUMBER_ORGANISMS; ++organisms)
	{
		for (genes = 0; genes  < NUMBER_GENES; ++genes)
		{
			m_currentGeneration[organisms][genes] = GetRandomGenome();
		}
	}

	// Initialize the model organisms
	for (genes = 0; genes < NUMBER_GENES; genes++)
	{
		m_modelOrganism[genes] = GetRandomGenome();
	}
}

bool EvaluateOrganisms()
{
	int organisms;
	int genes;
	int currentOrganismsFitnessTally;
	m_totalOfFitnesses = 0;

	for (organisms = 0; organisms < NUMBER_ORGANISMS; organisms++)
	{
		// Each similar value add to the total fitness of the evaluated organism
		currentOrganismsFitnessTally = 0;
		for (genes = 0; genes < NUMBER_GENES; genes++)
		{
			if (m_currentGeneration[organisms][genes] == m_modelOrganism[genes])
			{
				currentOrganismsFitnessTally++;
			}
		}

		m_organismsFitness[organisms] = currentOrganismsFitnessTally;

		m_totalOfFitnesses += currentOrganismsFitnessTally;

		// Check for a perfect organism
		if (currentOrganismsFitnessTally == MAXIMUM_FITNESS)
		{
			return true;
		}

	}

	return false;
}

int SelectOneOrganism()
{
	int runningTotal = 0;
	int randomSelectPoint = rand() % (m_totalOfFitnesses + 1);

	for (int organisms = 0; organisms < NUMBER_ORGANISMS; organisms++)
	{
		runningTotal += m_organismsFitness[organisms];
		if (runningTotal >= randomSelectPoint) return organisms;
	}
}

void ProduceNextGeneration()
{
	int organisms;
	int genes;
	int parentOne;
	int parentTwo;
	int crossoverPoint;
	int mutateThisGene;

	for (organisms = 0; organisms < NUMBER_ORGANISMS; organisms++)
	{
		parentOne = SelectOneOrganism();
		parentTwo = SelectOneOrganism();
		crossoverPoint = rand() % NUMBER_GENES;

		for (genes = 0; genes < NUMBER_GENES; genes++)
		{
			// Copy over a single gene
			mutateThisGene = rand() % (int)(1.0 / MUTATION_RATE);

			if (mutateThisGene == 0)
			{
				// Make this gene a mutation
				m_nextGeneration[organisms][genes] = GetRandomGenome();
			}
			else
			{
				// Copy the gene from the parent
				if (genes < crossoverPoint)
				{
					m_nextGeneration[organisms][genes] = m_currentGeneration[parentOne][genes];
				}
				else
				{
					m_nextGeneration[organisms][genes] = m_currentGeneration[parentTwo][genes];
				}
			}
		}
	}

	// Copy the children from the next generation to this generation
	for (organisms = 0; organisms < NUMBER_ORGANISMS; organisms++)
	{
		for (genes = 0; genes < NUMBER_GENES; genes++)
		{
			m_currentGeneration[organisms][genes] = m_nextGeneration[organisms][genes];
		}
	}
}

void PrintStatus()
{

	for (int organisms = 0; organisms < NUMBER_ORGANISMS; organisms++)
	{
		
		for (int genes = 0; genes < NUMBER_GENES; genes++)
		{
			std::cout << m_currentGeneration[organisms][genes] << "[" << m_modelOrganism[genes] << "] ";
		}
		std::cout << "Accuracy(%): "<<(double)m_organismsFitness[organisms] / MAXIMUM_FITNESS << " ";
		std::cout << std::endl;
	}
}

int DoOneRun()
{
	int generations = 1;
	bool perfectGeneration = false;
	InitializeOrganisms();

	while (true)
	{
		perfectGeneration = EvaluateOrganisms();
		if (perfectGeneration == true) return generations;
		ProduceNextGeneration();
		generations++;
		PrintStatus();
	}

}

int main()
{

	int finalGeneration;
	AllocateMemory();
	finalGeneration = DoOneRun();

	std::cout << "The perfect generation was: " << finalGeneration << std::endl;
	std::cin.ignore();
	return 0;
}