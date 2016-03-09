package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Example is an implementation of Network.  Notice that an
// implementation must include input and output wires.
type Example struct {
	*Network
	x, y, out *Wire
}

func newExample() *Example {
	n := &Example{
		Network: &Network{}, //TODO(y): Call NetNetwork
		x:       &Wire{},
		y:       &Wire{},
		out:     nil,
	}
	n.out = n.sig(n.dot(n.x, n.y))
	return n
}

func (n *Example) run() {
	n.x.value = 1
	n.y.value = 2
	n.Forward()
}

func TestExample(t *testing.T) {
	deterministic = true
	n := newExample()
	n.run()
	assert.Equal(t, n.out.value, sig(4))
}
