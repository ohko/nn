package nn

import (
	"fmt"
	"log"
	"math"
	"nn/mnist"
	"testing"
)

// go test nn -run Test_开发 -v -count=1 -timeout=1h
func Test_开发(t *testing.T) {
	o := &NN{
		Name: "开发", Learn: 0.6, MinDiff: math.Pow(0.01, 2), Count: 1000,
		Data: []StData{
			{input: []float64{0.1, 0.2}, output: []float64{0.3, 0.4}},
		},
		InputNum: 2, OutputNum: 1,
		Layer: []int{3, 3},
		Test: []StData{
			{input: []float64{0.3, 0.3}, output: []float64{0.6, 0.7}},
			{input: []float64{0.1, 0.1}, output: []float64{0.9, 0.8}},
		},
		Weight: [][][]float64{
			{[]float64{0.2, 0.3, 0.4}, []float64{0.5, 0.6, 0.7}},
			{[]float64{0.1, 0.2, 0.3}, []float64{0.4, 0.5, 0.6}, []float64{0.7, 0.8, 0.9}},
			{[]float64{0.2, 0.4}, []float64{0.6, 0.8}, []float64{0.3, 0.5}},
		},
	}

	if err := o.Train(); err != nil {
		t.Fatal(err)
	}
	o.Check(true, true)

	o.SaveWeight("aaat.weight")
}

// go test nn -run Test_教程样本 -v -count=1 -timeout=1h
func Test_教程样本(t *testing.T) {
	o := &NN{
		Name: "教程样本", Learn: 0.6, MinDiff: math.Pow(0.01, 2), Count: 1000,
		Data: []StData{
			{input: []float64{0.4, -0.7}, output: []float64{0.1}},
			{input: []float64{0.3, -0.5}, output: []float64{0.05}},
			{input: []float64{0.6, 0.1}, output: []float64{0.3}},
			{input: []float64{0.2, 0.4}, output: []float64{0.25}},
		},
		InputNum: 2, OutputNum: 1,
		Layer: []int{2},
		Test: []StData{
			{input: []float64{0.1, -0.2}, output: []float64{0.12}},
		},
		Weight: [][][]float64{
			{[]float64{0.1, 0.4}, []float64{-0.2, 0.2}},
			{[]float64{0.2}, []float64{-0.5}},
		},
	}

	if err := o.Train(); err != nil {
		t.Fatal(err)
	}
	o.Check(true, true)
}

// go test nn -run Test_01 -v -count=1 -timeout=1h
func Test_01(t *testing.T) {
	o := &NN{
		Name: "1", Learn: 0.6, MinDiff: math.Pow(0.1, 2), Count: 100000,
		InputNum: 2, OutputNum: 1,
		Data: []StData{
			{input: []float64{0, 0}, output: []float64{0}},
			{input: []float64{0, 1}, output: []float64{1}},
			{input: []float64{1, 0}, output: []float64{1}},
			{input: []float64{1, 1}, output: []float64{0}},
		},
		Layer: []int{6},
		Test: []StData{
			{input: []float64{0, 0}, output: []float64{0}},
			{input: []float64{0, 1}, output: []float64{1}},
			{input: []float64{1, 0}, output: []float64{1}},
			{input: []float64{1, 1}, output: []float64{0}},
		},
	}

	if err := o.Train(); err != nil {
		t.Fatal(err)
	}
	o.Check(true, true)
}

// go test nn -run Test_加法 -v -count=1 -timeout=1h
func Test_加法(t *testing.T) {
	o := &NN{
		Name: "加法", Learn: 0.6, MinDiff: math.Pow(0.03, 2), Count: 100000,
		InputNum: 2, OutputNum: 1,
		Layer: []int{3, 3, 3},
	}

	for i := 0; i < 1000; i++ {
		x, y := o.randFloat64(0.1, 0.49), o.randFloat64(0.1, 0.49)
		o.Data = append(o.Data, StData{input: []float64{x, y}, output: []float64{x + y}})
	}

	for i := 0; i < 1000; i++ {
		x, y := o.randFloat64(0.1, 0.49), o.randFloat64(0.1, 0.49)
		o.Test = append(o.Test, StData{input: []float64{x, y}, output: []float64{x + y}})
	}

	o.Train()
	o.Check(true, true)
}

// go test nn -run Test_Mnist -v -count=1 -timeout=1h
func Test_Mnist(t *testing.T) {
	o := &NN{
		Name: "MNIST", Learn: 0.6, MinDiff: math.Pow(0.01, 2), Count: 3,
		InputNum: 28 * 28, OutputNum: 10,
		Data:  []StData{},
		Layer: []int{5, 5},
	}

	{ // 训练数据
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
					bits[pos] = float64(vvv) / 0xff
					pos++
				}
			}
			out := make([]float64, 10)
			out[v.Digit] = 1
			o.Data = append(o.Data, StData{input: bits, output: out})
		}
	}

	{ // 测试数据
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
					bits[pos] = float64(vvv) / 0xff
					pos++
				}
			}
			out := make([]float64, 10)
			out[v.Digit] = 1
			o.Test = append(o.Test, StData{input: bits, output: out})
		}
	}

	max := func(data []float64) (int, float64) {
		mi, mv := 0, 0.0
		for k, v := range data {
			if v > mv {
				mi = k
				mv = v
			}
		}
		return mi, mv
	}
	o.CheckCallback = func(showLog bool, showPercent bool) float64 {
		chk := 0
		success := 0
		b := 0.0
		for _, v := range o.Test {
			chk++
			output := o.Right(v.input)
			ri, rv := max(output)
			ci, cv := max(v.output)

			if ri == ci {
				success++
			}
			if showLog {
				fmt.Printf("\r检测:%v | 期望：[%d]%0.8f | 结果：[%d]%0.8f | 误差：%0.8f   ", ri == ci, ci, cv, ri, rv, b)
			}
		}
		percent := float64(success) / float64(chk)
		if showPercent {
			fmt.Printf("\n检测:%v | 成功：%v | 成功率:%0.2f%%\n", chk, success, percent*100)
		}
		return percent
	}

	// weight
	if false {
		o.LoadWeight("plot/mnist.weight")
		o.Check(true, true)
		return
	}
	defer o.SaveWeight("plot/mnist.weight")

	if err := o.Train(); err != nil {
		t.Fatal(err)
	}

	o.Check(true, true)
}

func Test_sigmoid(t *testing.T) {
	log.Println(sigmoid([]float64{
		0, 1, 10, -10,
		0.001, 0.1, 0.9, 0.99, 1,
		-0.001, -0.1, -0.9, -0.999, -1}))
}

func Test_randFloat64(t *testing.T) {
	for i := 0; i < 100; i++ {
		log.Println((&NN{Seed: true}).randFloat64(0.1, 0.9))
	}
}
