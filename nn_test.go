package nn

import (
	"math"
	"testing"
)

// go Test nn -run Test_p -v -count=1
func Test_p(t *testing.T) {
	// p开发().Run()
	// p教程样本().Run()
	// p加法().Run()
	p乘法().Run()
}

func p开发() *NN {
	return &NN{
		Name: "开发", Learn: 0.6, MinDiff: 0.0001, Count: 1000,
		Data: []StData{
			StData{input: []float64{0.1, 0.2}, output: []float64{0.3, 0.4}},
		},
		Layer: []int{3, 3},
		Test: []StData{
			StData{input: []float64{0.3, 0.3}, output: []float64{0.6}},
			StData{input: []float64{0.1, 0.1}, output: []float64{0.9}},
		},
		Weight: [][][]float64{
			[][]float64{[]float64{0.2, 0.3, 0.4}, []float64{0.5, 0.6, 0.7}},
			[][]float64{[]float64{0.1, 0.2, 0.3}, []float64{0.4, 0.5, 0.6}, []float64{0.7, 0.8, 0.9}},
			[][]float64{[]float64{0.2, 0.4}, []float64{0.6, 0.8}, []float64{0.3, 0.5}},
		},
	}
}

func p教程样本() *NN {
	return &NN{
		Name: "教程样本", Learn: 0.6, MinDiff: 0.0001, Count: 1000,
		Data: []StData{
			StData{input: []float64{0.4, -0.7}, output: []float64{0.1}},
			StData{input: []float64{0.3, -0.5}, output: []float64{0.05}},
			StData{input: []float64{0.6, 0.1}, output: []float64{0.3}},
			StData{input: []float64{0.2, 0.4}, output: []float64{0.25}},
		},
		Layer: []int{2},
		Test: []StData{
			StData{input: []float64{0.1, -0.2}, output: []float64{0.12}},
		},
		Weight: [][][]float64{
			[][]float64{[]float64{0.1, 0.4}, []float64{-0.2, 0.2}},
			[][]float64{[]float64{0.2}, []float64{-0.5}},
		},
	}
}

func p乘法() *NN {
	o := &NN{
		Name: "乘法", Learn: 0.6, MinDiff: math.Pow(0.001, 2), Count: 1000,
		Data:  []StData{},
		Layer: []int{40, 20},

		Test: []StData{
			StData{input: []float64{0.9, 0.9}, output: []float64{0.81}},
			StData{input: []float64{0.1, 0.1}, output: []float64{0.01}},
		},
	}

	for i := 0; i < 1000; i++ {
		x, y := o.randFloat64(), o.randFloat64()
		o.Data = append(o.Data, StData{input: []float64{x, y}, output: []float64{x * y}})
	}

	return o
}

func p加法() *NN {
	o := &NN{
		Name: "加法", Learn: 0.6, MinDiff: math.Pow(0.001, 2), Count: 1000,
		Data:  []StData{},
		Layer: []int{40, 20},

		Test: []StData{
			StData{input: []float64{0.3, 0.3}, output: []float64{0.6}},
			StData{input: []float64{0.1, 0.1}, output: []float64{0.2}},
		},
	}

	for i := 0; i < 2000; i++ {
		x, y := o.randFloat64()/2, o.randFloat64()/2
		o.Data = append(o.Data, StData{input: []float64{x, y}, output: []float64{x + y}})
	}

	return o
}