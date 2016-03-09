package nn

import (
	"math"
	"math/rand"

	"github.com/wangkuiyi/parallel"
)

type Wire struct {
	value, grad float64
}

type Gate struct {
	ins      []*Wire     // Input wires. Refer to output wires owned by previous gates.
	ps       []Wire      // Parameters. Owned by the gate.
	out      Wire        // The output. Owned by the gate.
	forward  func(*Gate) // Should write to out.value.
	backward func(*Gate) // Should accumulate to ins.grad and ps.grad.
}

type Network struct {
	wireLayer map[*Wire]int // Map from *Wire to wire layer.  Used to infer gate layers.
	layers    [][]*Gate     // Layers of gates.
	outs      []*Wire       // Collected by Network.Outputs.
}

// Network.Register is supposed to be called by gate constructors,
// which should return &(gate.out).  We make Network.Register returns
// &(gate.out), so that its caller can simply return its return value.
func (n *Network) Register(gate *Gate) *Wire {
	if n.wireLayer == nil {
		n.wireLayer = make(map[*Wire]int)
	}

	// Find the up-most layer of gate inputs as the gate's layer.
	gateLayer := 0
	for _, in := range gate.ins {
		if _, ok := n.wireLayer[in]; !ok {
			// We assume that Go evaluates nested calls of
			// gate constructors in the bottom-first
			// order.  So if a gate input wire is not yet
			// registered, it must be a network input.
			n.wireLayer[in] = 0
		}

		l := n.wireLayer[in]
		if l > gateLayer {
			gateLayer = l
		}
	}

	// Register gate output wire in layer of gate layer + 1, so
	// upper layer gates can use this to infer their layer.
	n.wireLayer[&(gate.out)] = gateLayer + 1

	// Register gate's layer.
	if len(n.layers) < gateLayer+1 {
		n.layers = append(n.layers, make([]*Gate, 0))
	}
	n.layers[gateLayer] = append(n.layers[gateLayer], gate)

	return &(gate.out)
}

// Network.InferOutputs is supposed to be called after the
// construction of the network, which calls Network.Register to
// register wires and gates into Network.wireLayers and
// Network.layers, which in turn can be used by Network.InferOutputs
// to infer network output wires.
func (n *Network) InferOutputs() {
	n.outs = nil
	topWireLayer := len(n.layers)
	for k, v := range n.wireLayer {
		if v == topWireLayer {
			n.outs = append(n.outs, k)
		}
	}
}

func (n *Network) sig(ins ...*Wire) *Wire {
	gate := &Gate{
		ins: ins,
		ps:  nil, // sigmoid gate doesn't have parameters.
		out: Wire{},
		forward: func(gate *Gate) {
			gate.out.value = sig(gate.ins[0].value)
		},
		backward: func(gate *Gate) {
			s := sig(gate.ins[0].value)
			gate.ins[0].grad += (s * (1 - s)) * gate.out.grad
		},
	}
	return n.Register(gate)
}

func (n *Network) dot(ins ...*Wire) *Wire {
	gate := &Gate{
		ins: ins,
		ps:  randomWires(len(ins) + 1),
		out: Wire{},
		forward: func(gate *Gate) {
			gate.out.value = 0
			for i := 0; i < len(gate.ins); i++ {
				gate.out.value += gate.ins[i].value * gate.ps[i].value
			}
			gate.out.value += gate.ps[len(gate.ins)].value
		},
		backward: func(gate *Gate) {
			for i := 0; i < len(gate.ins); i++ {
				gate.ins[i].grad += gate.ps[i].value * gate.out.grad
				gate.ps[i].grad += gate.ins[i].value * gate.out.grad
			}
			gate.ps[len(gate.ins)].grad += gate.out.grad
		},
	}
	return n.Register(gate)
}

var (
	deterministic bool // Unit test and other debug-purposed code should set this to true.
)

func randomWires(n int) []Wire {
	r := make([]Wire, n)
	for i := range r {
		if deterministic {
			r[i].value = 1.0
		} else {
			r[i].value = rand.Float64()
		}
	}
	return r
}

func sig(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (n *Network) Forward() []*Wire {
	for l := range n.layers {
		parallel.For(0, len(n.layers[l]), 1, func(g int) {
			gate := n.layers[l][g]
			gate.forward(gate)
		})
	}
	return n.outs
}

type Example struct {
	*Network
	x, y, out *Wire
}

func NewExample() *Example {
	n := &Example{
		Network: &Network{}, //TODO(y): Call NetNetwork
		x:       &Wire{},
		y:       &Wire{},
		out:     nil,
	}
	n.out = n.sig(n.dot(n.x, n.y))
	n.InferOutputs()
	return n
}

func (n *Example) Run() {
	n.x.value = 1
	n.y.value = 2
	n.Forward()
}
