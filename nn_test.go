package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestForward(t *testing.T) {
	deterministic = true // Deterministic initialization of parameters.

	x := &Wire{}
	y := &Wire{}

	{
		n := &Network{}
		out := n.sig(n.dot(x, y))

		x.value = 1
		y.value = 2
		n.Forward()

		assert.Equal(t, out.value, sig(1*1+1*2+1))
	}
	{
		n := &Network{}
		out := n.sig(n.dot(
			n.sig(n.dot(x, y)),
			n.sig(n.dot(x, y))))

		n.Forward()

		assert.Equal(t, out.value, sig(
			sig(1*1+2*1+1)+
				sig(1*1+2*1+1)+
				1))
	}
}
