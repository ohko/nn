package nn

import (
	"log"
	"math"
	"nn/mnist"
	"testing"
)

// go test nn -run Test_p -v -count=1
func Test_p(t *testing.T) {
	// p开发()
	// p教程样本()
	// p乘法()
	// p加法()
	pMnist()
}

func p开发() {
	o := &NN{
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

	o.Train()
	o.Check(nil)
}

func p教程样本() {
	o := &NN{
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

	o.Train()
	o.Check(nil)
}

func p乘法() {
	o := &NN{
		Name: "乘法", Learn: 0.6, MinDiff: math.Pow(0.1, 2), Count: 100,
		Data:  []StData{},
		Layer: []int{10, 10},

		Test: []StData{
			StData{input: []float64{0.1, 0.1}, output: []float64{0.01}},
			StData{input: []float64{0.2, 0.2}, output: []float64{0.04}},
			StData{input: []float64{0.3, 0.3}, output: []float64{0.09}},
			StData{input: []float64{0.4, 0.4}, output: []float64{0.16}},
			StData{input: []float64{0.5, 0.5}, output: []float64{0.25}},
			StData{input: []float64{0.6, 0.6}, output: []float64{0.36}},
			StData{input: []float64{0.7, 0.7}, output: []float64{0.49}},
			StData{input: []float64{0.8, 0.8}, output: []float64{0.64}},
			StData{input: []float64{0.9, 0.9}, output: []float64{0.81}},
		},
	}

	for i := 0; i < 100000; i++ {
		x, y := o.randFloat64(), o.randFloat64()
		o.Data = append(o.Data, StData{input: []float64{x, y}, output: []float64{x * y}})
	}

	o.Train()
	o.Check(nil)
}

func p加法() {
	o := &NN{
		Name: "加法", Learn: 0.6, MinDiff: math.Pow(0.01, 2), Count: 1000,
		Data:  []StData{},
		Layer: []int{20, 20},

		Test: []StData{
			StData{input: []float64{0.3, 0.3}, output: []float64{0.6}},
			StData{input: []float64{0.1, 0.1}, output: []float64{0.2}},
		},
	}

	for i := 0; i < 2000; i++ {
		x, y := o.randFloat64()/2, o.randFloat64()/2
		o.Data = append(o.Data, StData{input: []float64{x, y}, output: []float64{x + y}})
	}

	o.Train()
	o.Check(nil)
}

func pMnist() {
	o := &NN{
		Name: "MNIST", Learn: 0.6, MinDiff: 0.01, Count: 3, Seed: false,
		Data:  []StData{},
		Layer: []int{20, 20},
	}

	{
		dataSet, err := mnist.ReadTrainSet("./mnist/MNIST_data")
		if err != nil {
			log.Fatal(err)
		}

		log.Printf("MNISST train: N:%v | W:%v | H:%v", dataSet.N, dataSet.W, dataSet.H)

		for _, v := range dataSet.Data {
			bits := make([]float64, dataSet.W*dataSet.H)
			pos := 0
			for _, vv := range v.Image {
				for _, vvv := range vv {
					if vvv > 0 {
						bits[pos] = 0.9
					} else {
						bits[pos] = 0.1
					}
					pos++
				}
			}
			o.Data = append(o.Data, StData{input: bits, output: []float64{(float64(v.Digit) + 0.001) / 10}})
		}
	}
	{
		dataSet, err := mnist.ReadTestSet("./mnist/MNIST_data")
		if err != nil {
			log.Fatal(err)
		}

		log.Printf("MNISST test: N:%v | W:%v | H:%v", dataSet.N, dataSet.W, dataSet.H)

		for _, v := range dataSet.Data {
			bits := make([]float64, dataSet.W*dataSet.H)
			pos := 0
			for _, vv := range v.Image {
				for _, vvv := range vv {
					if vvv > 0 {
						bits[pos] = 0.9
					} else {
						bits[pos] = 0.1
					}
					pos++
				}
			}
			o.Test = append(o.Test, StData{input: bits, output: []float64{float64(v.Digit)}})
		}
	}

	o.Train()
	o.Check(func(chk, result float64) bool {
		return math.Round(chk*10) == result
	})
}
