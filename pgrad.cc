#include <iostream>
#include <functional>
#include <random>
#include <iomanip>
#include <cassert>
#include <vector>

#define PRINT_VEC(vec)                           \
	do                                           \
	{                                            \
		if (vec.size() == 1)                     \
		{                                        \
			std::cout << *(vec[0]) << std::endl; \
		}                                        \
		else                                     \
		{                                        \
			for (const auto &elem : vec)         \
			{                                    \
				std::cout << *elem << "\n";      \
			}                                    \
			std::cout << std::endl;              \
		}                                        \
	} while (0)
#define VECTOR_VAL_TYPE std::vector<std::shared_ptr<Value>>
#define VAL(x) std::make_shared<Value>(x)

double gen_random_weight()
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	static std::uniform_real_distribution<double> dis(-1.0, 1.0);
	return dis(gen);
}

class Value : public std::enable_shared_from_this<Value>
{
public:
	Value(double data, std::vector<std::shared_ptr<Value>> prev = {}, std::string op = "")
	{
		this->m_data = data;
		this->m_grad = 0.0;
		this->m_prev = std::move(prev);
		this->m_op = op;
		this->_backward = [] {};
	}

	friend std::ostream &operator<<(std::ostream &ostr, Value &val)
	{
		ostr << std::setprecision(4) << std::fixed;
		ostr << "Value(data=" << val.get_data() << ", grad=" << val.get_grad() << ", op=\"" << val.get_op() << "\", label=\"" << val.get_label() << "\", prev=" << val.m_prev.size() << ")";

		return ostr;
	}

	std::shared_ptr<Value> pow(std::shared_ptr<Value> val)
	{
		auto out = std::make_shared<Value>(std::pow(m_data, val->get_data()), std::vector<std::shared_ptr<Value>>{shared_from_this(), val}, "**");
		out->_backward = [out, self = shared_from_this(), val]()
		{
			self->inc_grad(val->get_data() * std::pow(self->get_data(), (val->get_data() - 1)) * out->get_grad());
		};
		return out;
	}

	std::shared_ptr<Value> pow(double val)
	{
		return pow(std::make_shared<Value>(val));
	}

	void backward()
	{
		m_grad = 1.0;
		auto topo_order = build_topo();
		for (auto val : topo_order)
		{
			val->_backward();
		}
	}

	std::vector<std::shared_ptr<Value>> build_topo()
	{
		std::vector<std::shared_ptr<Value>> topo_order;
		build_topo(topo_order);
		clear_visit_mark(topo_order);
		return topo_order;
	}

	void clear_visit_mark(std::vector<std::shared_ptr<Value>> &topo_order)
	{
		for (auto &val : topo_order)
		{
			val->set_visited(false);
		}
	}

	void build_topo(std::shared_ptr<Value> val, std::vector<std::shared_ptr<Value>> &topo_order)
	{
		if (!val->get_visited())
		{
			val->set_visited(true);
			for (auto child : val->get_prev())
			{
				build_topo(child, topo_order);
			}
			topo_order.insert(topo_order.begin(), val);
		}
	}

	void build_topo(std::vector<std::shared_ptr<Value>> &topo_order)
	{
		build_topo(shared_from_this(), topo_order);
	}

	std::vector<std::shared_ptr<Value>> get_prev()
	{
		return m_prev;
	}

	double get_data()
	{
		return m_data;
	}

	void set_data(double val)
	{
		m_data = val;
	}

	void inc_data(double val)
	{
		m_data += val;
	}

	std::string get_op()
	{
		return m_op;
	}

	void set_label(std::string label)
	{
		m_label = label;
	}

	std::string get_label()
	{
		return m_label;
	}

	void set_grad(double grad)
	{
		m_grad = grad;
	}

	void inc_grad(double grad)
	{
		m_grad += grad;
	}

	double get_grad()
	{
		return m_grad;
	}

	bool get_visited()
	{
		return m_visited;
	}

	void set_visited(bool val)
	{
		m_visited = val;
	}

	std::function<void()> _backward;

private:
	double m_data;
	double m_grad;
	bool m_visited;
	std::vector<std::shared_ptr<Value>> m_prev;
	std::string m_op;
	std::string m_label;
};

std::shared_ptr<Value> operator+(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
	auto prev = std::vector<std::shared_ptr<Value>>{lhs, rhs};
	auto out = std::make_shared<Value>(lhs->get_data() + rhs->get_data(), prev, "+");
	out->_backward = [out, lhs, rhs]()
	{
		// std::cout << "lhs :: " << *lhs << "\n";
		// std::cout << "rhs :: " << *rhs << "\n";
		lhs->inc_grad(1.0 * out->get_grad());
		rhs->inc_grad(1.0 * out->get_grad());
	};
	return out;
}

std::shared_ptr<Value> operator*(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
	auto prev = std::vector<std::shared_ptr<Value>>{lhs, rhs};
	auto out = std::make_shared<Value>(lhs->get_data() * rhs->get_data(), prev, "*");
	out->_backward = [out, lhs, rhs]()
	{
		lhs->inc_grad(rhs->get_data() * out->get_grad());
		rhs->inc_grad(lhs->get_data() * out->get_grad());
	};
	return out;
}

// multi-directional overloading for ops
// add op
std::shared_ptr<Value> operator+(std::shared_ptr<Value> lhs, double rhs)
{
	return lhs + std::make_shared<Value>(rhs);
}
std::shared_ptr<Value> operator+(double lhs, std::shared_ptr<Value> rhs)
{
	return rhs + lhs;
}

// mul op
std::shared_ptr<Value> operator*(std::shared_ptr<Value> lhs, double rhs)
{
	return lhs * std::make_shared<Value>(rhs);
}
std::shared_ptr<Value> operator*(double lhs, std::shared_ptr<Value> rhs)
{
	return rhs * lhs;
}

// div op
std::shared_ptr<Value> operator/(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
	return lhs * rhs->pow(-1);
}
std::shared_ptr<Value> operator/(std::shared_ptr<Value> lhs, double rhs)
{
	return lhs / std::make_shared<Value>(rhs);
}
std::shared_ptr<Value> operator/(double lhs, std::shared_ptr<Value> rhs)
{
	return std::make_shared<Value>(lhs) / rhs;
}

// negate
std::shared_ptr<Value> operator-(std::shared_ptr<Value> rhs)
{
	return rhs * std::make_shared<Value>(-1.0);
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs)
{
	return lhs + (-rhs);
}

std::shared_ptr<Value> tanh(std::shared_ptr<Value> lhs)
{
	auto t = std::tanh(lhs->get_data());
	auto prev = std::vector<std::shared_ptr<Value>>{lhs};
	auto out = std::make_shared<Value>(t, prev, "tanh");
	out->_backward = [out, lhs]()
	{
		lhs->inc_grad((1 - pow(out->get_data(), 2)) * out->get_grad());
	};
	return out;
}

// NEURON IMPLEMENTATION
class Neuron
{
public:
	Neuron(int nin)
	{
		this->m_weights.reserve(nin);
		for (size_t i = 0; (int)i < nin; i++)
		{
			auto weight = std::make_shared<Value>(gen_random_weight());
			this->m_weights.emplace_back(weight);
		}
	}

	std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>> &x)
	{
		std::shared_ptr<Value> act = std::make_shared<Value>(0.0);
		for (size_t i = 0; i < x.size(); i++)
		{
			act = act + (x[i] * this->m_weights[i]);
		}
		act = act + this->m_bias;
		auto out = tanh(act);
		return out;
	}

	void print_params()
	{
		std::cout << "weights: ";
		for (auto &weight : this->m_weights)
		{
			std::cout << weight->get_data() << ",";
		}
		std::cout << "\nbias: " << this->m_bias->get_data() << "\n";
	}

	std::vector<std::shared_ptr<Value>> parameters()
	{
		std::vector<std::shared_ptr<Value>> params;
		params.reserve(this->m_weights.size() + 1);

		for (size_t i = 0; i < this->m_weights.size(); i++)
		{
			params.emplace_back(this->m_weights[i]);
		}
		params.emplace_back(this->m_bias);
		return params;
	}

private:
	std::vector<std::shared_ptr<Value>> m_weights;
	std::shared_ptr<Value> m_bias = std::make_shared<Value>(gen_random_weight());
};

class Layer
{
public:
	Layer(int nin, int nout) : m_nin(nin), m_nout(nout)
	{
		this->m_neurons.reserve(nout);
		for (size_t i = 0; (int)i < nout; i++)
		{
			auto n = Neuron(nin);
			this->m_neurons.emplace_back(n);
		}
	}

	friend std::ostream &operator<<(std::ostream &ostr, Layer &n)
	{
		ostr << "Layer(nin=" << n.get_nin() << ", nout=" << n.get_nout() << ")";
		return ostr;
	}

	std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> &x)
	{
		std::vector<std::shared_ptr<Value>> outs;
		outs.reserve(this->m_neurons.size() + 1);

		for (auto &neuron : this->m_neurons)
		{
			outs.emplace_back(neuron(x));
		}
		return outs;
	}

	std::vector<std::shared_ptr<Value>> parameters()
	{
		std::vector<std::shared_ptr<Value>> params;
		auto total_params = (this->m_nin + 1) * this->m_nout;
		params.reserve(total_params + 1);

		for (auto neuron : this->m_neurons)
		{
			for (auto weight : neuron.parameters())
			{
				params.emplace_back(weight);
			}
		}
		return params;
	}

	void print_params()
	{
		auto total_params = (this->m_nin + 1) * this->m_nout;
		std::cout << "Layer parameters: " << total_params << std::endl;
		for (auto neuron : this->m_neurons)
		{
			neuron.print_params();
		}
	}

	int get_nin()
	{
		return m_nin;
	}

	int get_nout()
	{
		return m_nout;
	}

private:
	int m_nin;
	int m_nout;
	std::vector<Neuron> m_neurons;
};

class MLP
{
public:
	MLP(int nin, std::vector<int> nout)
	{
		std::vector<int> sz;
		sz.reserve(nout.size() + 1);

		this->m_layers.reserve(nout.size() + 1);
		sz.emplace_back(nin);

		for (int out : nout)
		{
			sz.emplace_back(out);
		}

		for (size_t i = 0; i < nout.size(); i++)
		{
			this->m_layers.emplace_back(Layer(sz[i], sz[i + 1]));
		}
	}

	std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x)
	{
		for (auto layer : m_layers)
		{
			x = layer(x);
		}
		return x;
	}

	std::vector<std::shared_ptr<Value>> parameters()
	{
		std::vector<std::shared_ptr<Value>> params;
		params.reserve(m_total_params + 1);

		for (auto layer : m_layers)
		{
			for (auto param : layer.parameters())
			{
				params.emplace_back(param);
			}
		}

		return params;
	}

	void print_params()
	{
		for (int i = 0; i < (int)this->m_layers.size(); i++)
		{
			std::cout << "\nLayer " << i + 1 << ": " << std::endl;
			this->m_layers[i].print_params();
		}
	}

	void zero_grad()
	{
		for (auto &weight : parameters())
		{
			weight->set_grad(0.0);
		}
	}

private:
	int m_total_params;
	std::vector<Layer> m_layers;
};

std::shared_ptr<Value> calculate_loss(VECTOR_VAL_TYPE ys, std::vector<VECTOR_VAL_TYPE> ypred)
{
	auto acc = std::make_shared<Value>(0.0);
	assert((ys.size() == ypred.size() && "size of ys and ypred cannot be different"));

	for (int i = 0; i < (int)std::min(ys.size(), ypred.size()); ++i)
	{
		acc = acc + (ypred[i][0] - ys[i])->pow(2);
	}
	return acc;
}

auto main() -> int
{

	auto xs = std::vector<VECTOR_VAL_TYPE>{
		VECTOR_VAL_TYPE{VAL(2.0), VAL(3.0), VAL(-1.0)},
		VECTOR_VAL_TYPE{VAL(3.0), VAL(-1.0), VAL(0.5)},
		VECTOR_VAL_TYPE{VAL(0.5), VAL(1.0), VAL(1.0)},
		VECTOR_VAL_TYPE{VAL(1.0), VAL(1.0), VAL(-1.0)}};

	auto ys = VECTOR_VAL_TYPE{VAL(1.0), VAL(-1.0), VAL(-1.0), VAL(1.0)};
	auto mlp = MLP(3, std::vector<int>{4, 4, 1});
	std::shared_ptr<Value> loss;

	// 100,000 iterations result in a loss of 0.00000285361502291864
	for (size_t k = 0; k < 10; ++k)
	{
		std::vector<VECTOR_VAL_TYPE> ypred;
		ypred.reserve(xs.size() + 1);

		for (auto x : xs)
		{
			ypred.emplace_back(mlp(x));
		}
		loss = calculate_loss(ys, ypred);
		mlp.zero_grad();

		loss->backward();

		for (auto &weight : mlp.parameters())
		{
			weight->set_data(weight->get_data() - (0.1 * weight->get_grad()));
		}

		std::cout << k << ". " << loss->get_data() << "\n";

		for (auto &el : ypred)
		{
			for (auto &e : el)
			{
				std::cout << std::setprecision(20) << std::fixed;
				std::cout << e->get_data() << "\n";
			}
		}
	}

	return 0;
}