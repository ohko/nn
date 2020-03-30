package nn

import (
	"fmt"
	"log"
	"math"
	"time"

	mrand "math/rand"

	"github.com/ohko/logger"
)

// NN Neural Network
type NN struct {
	ll      *logger.Logger
	Seed    bool          // 随机前是否先seed
	Name    string        // 名称
	Learn   float64       // 学习率
	MinDiff float64       // 最小误差
	Count   int           // 训练次数
	Data    []StData      // 输入/输出层
	Layer   []int         // 隐藏层数量
	Hidden  [][]float64   // 隐藏层
	Weight  [][][]float64 // 权重
	Output  []float64     // 输出层
	Test    []StData      // 测试
}

// StData ...
type StData struct {
	input  []float64
	output []float64
}

func (o *NN) sigmoid(x []float64) []float64 {
	for k, v := range x {
		x[k] = 1 / (1 + math.Exp(-v))
		// x[k] = (1 - math.Exp(-2*v)) / (1 + math.Exp(-2*v))
		// x[k] = 1 - math.Pow(v, 2)
		// x[k] = math.Sinh(v) / math.Cosh(v)
	}
	return x
}

func (o *NN) matrixMul(input []float64, weight [][]float64) []float64 {
	z := make([]float64, len(weight[0]))

	// o.ll.Log0Debug("input:", input)
	// o.ll.Log0Debug("weigth:", weight)

	for i := 0; i < len(weight[0]); i++ {
		for j := 0; j < len(input); j++ {
			x, y := input[j], weight[j][i]
			// o.ll.Log0Debug(fmt.Sprint(x, "x", y, "=>", i))
			z[i] += x * y
		}
	}

	// sigmoid激活
	z = o.sigmoid(z)
	// o.ll.Log0Debug("output:", z)
	return z
}

func (o *NN) right(data *StData) {
	output := data.input
	for index := 0; index < len(o.Weight); index++ {
		// 输入层加权求和
		output = o.matrixMul(output, o.Weight[index])
		// 保存
		if index < len(o.Weight)-1 {
			o.Hidden[index] = output
		} else {
			o.Output = output
		}
	}

	// o.ll.Log0Debug("last output:", o.Output)
}

func (o *NN) matrixMul2(input []float64, weightIndex int, layer []float64) []float64 {
	z := make([]float64, len(o.Weight[weightIndex]))

	// o.ll.Log0Debug("input:", input)
	// o.ll.Log0Debug("weigth:", o.Weight[weightIndex])
	// o.ll.Log0Debug("layer:", layer)

	for i := 0; i < len(o.Weight[weightIndex]); i++ {
		for j := 0; j < len(input); j++ {
			x, y, l := input[j], o.Weight[weightIndex][i][j], layer[i]
			// o.ll.Log0Debug(fmt.Sprint(x, "x", y, "x", l, "x(1-", l, ")=>z:", i))
			z[i] += x * y * l * (1 - l)
			// o.ll.Log0Debug(fmt.Sprint(y, " + ", l, " * ", x, " * ", o.Learn, "=>", y+l*x*o.Learn))
			o.Weight[weightIndex][i][j] = y + l*x*o.Learn
		}
	}

	// o.ll.Log0Debug("output:", z)
	// o.ll.Log0Debug("new weight:", o.Weight[weightIndex])

	return z
}

func (o *NN) left(data *StData) {
	rdiff := make([]float64, len(o.Output))
	// 计算残差
	for k := range data.output {
		rdiff[k] = -(o.Output[k] - data.output[k]) * o.Output[k] * (1 - o.Output[k])
	}
	// o.ll.Log0Debug("残差:", rdiff)

	// 修正每层残差
	output := rdiff
	for index := len(o.Weight) - 1; index >= 0; index-- {
		// 输入层加权求和
		if index == 0 {
			output = o.matrixMul2(output, index, data.input)
		} else {
			output = o.matrixMul2(output, index, o.Hidden[index-1])
		}
	}

	// o.ll.Log0Debug("weight:", o.Weight)
}

func (o *NN) init() {
	o.ll = logger.NewLogger(nil)
	o.ll.SetFlags(log.Lshortfile)

	if o.MinDiff <= 0 {
		o.MinDiff = 0.0001
	}
	if o.Count <= 0 {
		o.Count = 1000
	}

	fmt.Printf("name:%v | diff:%f | count:%v | layer:%v\n", o.Name, o.MinDiff, o.Count, o.Layer)

	// generate hidden layer
	for _, v := range o.Layer {
		o.Hidden = append(o.Hidden, make([]float64, v))
	}
	// o.ll.Log0Debug("hidden:", o.Hidden)

	// generate weight
	if o.Weight == nil {
		tmp := []int{len(o.Data[0].input)}
		tmp = append(tmp, o.Layer...)
		tmp = append(tmp, len(o.Data[0].output))
		for i := 0; i < len(tmp)-1; i++ {
			t1 := make([][]float64, tmp[i])
			for j := 0; j < tmp[i]; j++ {
				t1[j] = make([]float64, tmp[i+1])
				for m := 0; m < len(t1[j]); m++ {
					if o.randFloat64() < 0.5 {
						t1[j][m] = o.randFloat64()
					} else {
						t1[j][m] = -o.randFloat64()
					}
				}
			}
			o.Weight = append(o.Weight, t1)
		}
	}
	// o.ll.Log4Trace("weight:", fmt.Sprintf("%0.2v", o.Weight))
}

// Train ...
func (o *NN) Train() {
	o.init()

	study := 0
	diff := make([]float64, len(o.Data[0].output))
	max := o.MinDiff + 1
	for count := 1; max > o.MinDiff && count <= o.Count; count++ {
		max = 0
		for k1 := range o.Data {
			study++
			o.train(&o.Data[k1], &diff)

			for _, d := range diff {
				if d > max {
					max = d
				}
			}

			if study%1000 == 0 {
				fmt.Printf("\r训练：%v/%v(%0.1f%%) | 误差：%0.8f", study, o.Count*len(o.Data), float64(study)/float64(o.Count*len(o.Data))*100, max)
			}
		}
	}
	fmt.Println()
	fmt.Println("学习次数:", study)

}

// Check ...
func (o *NN) Check(checkFun func(chk, result float64) bool) {
	chk := 0
	success := 0
	b := false
	for _, v := range o.Test {
		o.right(&v)
		for kk := range v.output {
			chk++
			if checkFun != nil {
				b = checkFun(o.Output[kk], v.output[kk])
			} else {
				b = o.defaultCheckFun(o.Output[kk], v.output[kk])
			}
			if b {
				success++
			}
			fmt.Printf("\r检测:%v | 期望：%v | 结果：%v   ", b, o.Output[kk], v.output[kk])
		}
	}
	fmt.Printf("\n检测:%v | 成功：%v | 成功率:%0.2f%%\n", chk, success, float64(success)/float64(chk)*100)
}

func (o NN) defaultCheckFun(chk, result float64) bool {
	diff := math.Pow(chk-result, 2)
	b := diff < o.MinDiff
	return b
}

func (o *NN) train(v *StData, diff *[]float64) {
	o.right(v)
	for kk := range v.output {
		(*diff)[kk] = math.Pow(o.Output[kk]-v.output[kk], 2)
	}
	o.left(v)
}

// mrand "math/rand"
func (o *NN) randFloat64() float64 {
	if o.Seed {
		o.Seed = false
		mrand.Seed(time.Now().UnixNano())
	}
	for {
		x := mrand.Float64()
		if x != 0 {
			return x
		}
	}
}

// fangfaGorman指出隐层结点数s与模式数N的关系是：s＝log2N；
// n为输入层结点数
func (o *NN) perLevelNode1(n int) int {
	return int(math.Ceil(math.Log2(float64(n))))
}

// Kolmogorov定理表明，隐层结点数s＝2n＋1
// n为输入层结点数
func (o *NN) perLevelNode2(n int) int {
	return 2*n + 1
}

// s＝sqrt（0.43mn＋0.12nn＋2.54m＋0.77n＋0.35）＋0.51
// m是输入层的个数，n是输出层的个数
func (o *NN) perLevelNode3(m, n int) int {
	return int(math.Ceil(math.Sqrt(0.43*float64(m)*float64(n)+0.12*float64(n)*float64(n)+2.54*float64(m)+0.77*float64(n)+0.35) + 0.51))
}
