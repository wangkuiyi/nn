package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExample(t *testing.T) {
	deterministic = true // Deterministic initialization of parameters.

	n := &Network{}
	x := &Wire{}
	y := &Wire{}

	out := n.sig(n.dot(x, y))

	// n.out = n.sig(n.dot(
	// 	n.sig(n.dot(n.x, n.y)),
	// 	n.sig(n.dot(n.x, n.y))))

	x.value = 1
	y.value = 2
	n.Forward()

	assert.Equal(t, out.value, sig(4))
}
