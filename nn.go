package main

import (
	"math"
	"math/rand"
)

type Wire struct {
	value, grad float64
}

type Gate struct {
	ins      []*Wire // Input wires.
	ps       []Wire  // Parameters.
	out      Wire
	forward  func(*Gate) // Write to out.value.
	backward func(*Gate) // Accumulate to ins.grad and ps.grad.
}

type Network struct {
	wireLayer map[*Wire]int // Map from *Wire to wire layer.
	layers    [][]*Gate     // Layers of gates.
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
	return n.register(gate)
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
	return n.register(gate)
}

func randomWires(n int) []Wire {
	r := make([]Wire, n)
	for i := range r {
		r[i].value = rand.Float64()
	}
	return r
}

func sig(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// register returns gate.out
func (n *Network) register(gate *Gate) *Wire {
	// TODO(y): register here.
	return &(gate.out)
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
	n.out = n.sig(n.dot(
		n.sig(n.dot(n.x, n.y)),
		n.sig(n.dot(n.x, n.y))))
	return n
}

func main() {}
