package kalman

import (
	"gonum.org/v1/gonum/mat"
)

func MakeIdentityMatrix(n int) *mat.Dense {
	m := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		m.Set(i, i, 1)
	}
	return m
}
