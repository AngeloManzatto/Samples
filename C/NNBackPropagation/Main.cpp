/*********************************************************************

Purpose: Simple Neural Network for solving XOR table using Back Propagation Learning

@author Angelo Antonio Manzatto
@version 1.0 31/08/2017

*********************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>
#include <string>

static std::vector<std::vector<double>> XOR_INPUT = {
	{ 0.0, 0.0 },
	{ 1.0, 0.0 },
	{ 0.0, 1.0 },
	{ 1.0, 1.0 },

};

static std::vector<std::vector<double>> XOR_TARGET = {
	{ 0.0 },
	{ 1.0 },
	{ 1.0 },
	{ 0.0 },

};

inline double GetRandomNumber()
{

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(-1, 1);

	return dis(gen);
}

inline double Activation(double x)
{
	if (x < -50.0)
	{
		return 0.0;
	}
	else if (x > 50.0)
	{
		return 1.0;
	}
	return 1.0 / (1.0 + std::exp(-1.0 * x));
}

inline double Derived(double x)
{
	return x * (1.0 - x);
}


/*************************************************/
struct Connection
{
	double weight;
	double prevDeltaWeight;
	Connection() :weight(GetRandomNumber()), prevDeltaWeight(0.0) {}
};

/*************************************************/
struct Neuron
{
	std::vector<Connection> connections;
	double value;
	double gradient;
	double bias;
	double prevDeltaBias;
	Neuron() :value(0.0), gradient(0.0), bias(GetRandomNumber()), prevDeltaBias(0.0) {}

	int GetConnectionsCount() { return connections.size(); }
};

/*************************************************/
struct Layer
{
	std::vector<Neuron> neurons;

	int GetNeuronCount() { return neurons.size(); }
	Layer(int neuronCount)
	{
		neurons.resize(neuronCount);
	}
};

/*************************************************/
class NeuralNetwork
{
public:
	NeuralNetwork(const std::vector<int>& topology, double learningRate = 0.9, double momentum = 0.7);
	~NeuralNetwork();

	// Getters
	int    GetLayerCount() { return m_layers.size(); }
	double GetError() { return m_error; }

	std::vector<double> FeedForward(const std::vector<double>& pattern);
	void Train(const std::vector<double>& pattern, const std::vector<double>& target);

	// Console
	void Print();

private:
	std::vector<Layer> m_layers;
	std::vector<int>   m_topology;

	double m_learningRate;
	double m_momentum;

	double m_error;

};

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology, double learningRate, double momentum)
{
	m_learningRate = learningRate;
	m_momentum = momentum;
	m_topology = topology;

	// Create Topology
	for (unsigned l = 0; l < topology.size(); l++)
	{
		// Create Layers
		Layer layer(topology[l]);


		// Create Neurons for each Layer
		for (int n = 0; n < layer.GetNeuronCount(); n++)
		{
			layer.neurons[n] = Neuron();
		}

		// Create Connections for each Neuron
		for (unsigned n = 0; n < layer.neurons.size(); n++)
		{
			if (l == 0)
			{
				layer.neurons[n].bias = 0;
			}
			else
			{
				for (unsigned d = 0; d < m_layers[l - 1].neurons.size(); d++)
				{
					layer.neurons[n].connections.push_back(Connection());
				}
			}
		}

		m_layers.push_back(layer);

	}

}

NeuralNetwork::~NeuralNetwork()
{
}

std::vector<double> NeuralNetwork::FeedForward(const std::vector<double>& pattern)
{
	assert(pattern.size() == m_layers[0].GetNeuronCount());

	for (unsigned l = 0; l < m_layers.size(); l++)
	{
		Layer &layer = m_layers[l];

		for (int n = 0; n < layer.GetNeuronCount(); n++)
		{
			Neuron &neuron = layer.neurons[n];

			// Process Input layer
			if (l == 0)
			{
				neuron.value = pattern[n];
			}
			else
			{
				// Process hidden layers and output
				neuron.value = 0.0;
				// Output = Activation(Sum(inputs * weights))
				for (int np = 0; np < m_layers[l - 1].GetNeuronCount(); np++)
				{
					neuron.value = neuron.value + m_layers[l - 1].neurons[np].value * neuron.connections[np].weight;
				}
				neuron.value = Activation(neuron.value + neuron.bias);
			}
		}
	}

	// Return result from output layers
	std::vector<double> result;
	Layer& outputLayer = m_layers.back();
	for (int n = 0; n < outputLayer.GetNeuronCount(); n++)
	{
		result.push_back(outputLayer.neurons[n].value);
	}

	return result;

}

// Train the Network on a pattern against a specific target
void NeuralNetwork::Train(const std::vector<double>& pattern, const std::vector<double>& target)
{
	assert(pattern.size() == m_layers[0].GetNeuronCount() && target.size() == m_layers.back().GetNeuronCount());

	// Process pattern
	FeedForward(pattern);

	Layer &outputLayer = m_layers.back();

	// Calculate Global error
	m_error = 0;
	for (int i = 0; i < outputLayer.GetNeuronCount(); i++)
	{
		double error = target[i] - outputLayer.neurons[i].value;
		m_error += error * error;
	}

	m_error /= outputLayer.GetNeuronCount();
	m_error = sqrt(m_error);

	// Caculate Delta for output layer

	for (int i = 0; i < outputLayer.GetNeuronCount(); i++)
	{

		// Calculate output gradient;
		Neuron &outputNeuron = outputLayer.neurons[i];
		outputNeuron.gradient = Derived(outputNeuron.value) * (target[i] - outputNeuron.value);

		// Calculate hidden gradient;
		for (unsigned j = m_layers.size() - 2; j > 0; j--)
		{
			for (int k = 0; k < m_layers[j].GetNeuronCount(); k++)
			{
				Neuron &neuron = m_layers[j].neurons[k];
				neuron.gradient = Derived(neuron.value) *
					m_layers[j + 1].neurons[i].connections[k].weight *
					m_layers[j + 1].neurons[i].gradient;
			}
		}

	}

	// Update Weights
	for (unsigned i = m_layers.size() - 1; i > 0; i--)
	{
		for (int j = 0; j < m_layers[i].GetNeuronCount(); j++)
		{
			Neuron &neuron = m_layers[i].neurons[j];
			double deltaBias = m_learningRate * neuron.gradient;
			neuron.bias += deltaBias + m_momentum * neuron.prevDeltaBias;
			neuron.prevDeltaBias = deltaBias;

			for (unsigned k = 0; k < neuron.connections.size(); k++)
			{
				double deltaWeight = m_learningRate * m_layers[i - 1].neurons[k].value * neuron.gradient;
				neuron.connections[k].weight += deltaWeight + m_momentum * neuron.connections[k].prevDeltaWeight;
				neuron.connections[k].prevDeltaWeight = deltaWeight;
			}
		}

	}

}

//Print the Network on Console
void NeuralNetwork::Print()
{
	for (int l = 0; l < GetLayerCount(); l++)
	{
		std::cout << "Layer[" << l << "]" << std::endl;

		for (int n = 0; n < m_layers[l].GetNeuronCount(); n++)
		{
			std::cout << " Neuron[" << n << "]: Value: " << m_layers[l].neurons[n].value
				<< " Bias: " << m_layers[l].neurons[n].bias
				<< " Gradient: " << m_layers[l].neurons[n].gradient
				<< std::endl;

			for (int c = 0; c < m_layers[l].neurons[n].GetConnectionsCount(); c++)
			{
				std::cout << "  Connection[" << c << "]: Weight: " << m_layers[l].neurons[n].connections[c].weight << std::endl;
			}
		}

		std::cout << std::endl;
	}
}

/*************************************************/
int main()
{

	std::vector<int> topology = { 2,3,1 };
	NeuralNetwork nn(topology);

	int epochs = 1;
	double averageError = 0;

	// Train the network
	do
	{
		for (unsigned i = 0; i < XOR_INPUT.size(); i++)
		{
			nn.Train(XOR_INPUT[i], XOR_TARGET[i]);
			averageError += nn.GetError();
		}
		averageError /= XOR_INPUT.size();

		std::cout << "Error: " << averageError << " Epoch: " << epochs << std::endl;
		epochs++;
	} while (averageError > 0.01);

	// Test and Validate the Network
	std::cout << "\n=============Test Network===============" << std::endl;
	std::cout << "=============XOR TABLE===============" << std::endl;

	for (unsigned i = 0; i < XOR_INPUT.size(); i++)
	{
		std::vector<double> result = nn.FeedForward(XOR_INPUT[i]);

		std::cout << "Input Data:{" << XOR_INPUT[i][0] << "," << XOR_INPUT[i][1] << "} Target: " << XOR_TARGET[i][0] << " Result: " << result[0] << std::endl;
	}

	std::cin.ignore();
	return 0;
}