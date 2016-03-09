package nn

import (
	"fmt"
	"testing"

	"github.com/davecgh/go-spew/spew"
)

func TestExample(t *testing.T) {
	deterministic = true
	n := NewExample()
	// spew.Dump(n)
	n.Run()
	spew.Dump(n.outs)

	fmt.Printf("sig(4) = %f", sig(4))
}
