package nn

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"runtime"
	"time"

	mrand "math/rand"

	"github.com/ohko/logger"
)

// NN Neural Network
type NN struct {
	ll                 *logger.Logger
	Seed               bool     // 随机前是否先seed
	Name               string   // 名称
	Learn              float64  // 学习率
	MinDiff            float64  // 最小误差
	Count              int      // 训练次数
	Data               []StData // 输入/输出层
	InputNum           int
	OutputNum          int
	Layer              []int                             // 隐藏层数量
	Hidden             [][]float64                       // 隐藏层
	Weight             [][][]float64                     // 权重
	Output             []float64                         // 输出层
	Test               []StData                          // 测试
	TestCallback       func(chk, result float64) float64 // 检测回调函数
	StudyCountCallback func(study int)                   // 学习次数回调
}

// StData ...
type StData struct {
	input  []float64
	output []float64
}

func sigmoid(x []float64) []float64 {
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
	z = sigmoid(z)
	// o.ll.Log0Debug("output:", z)
	return z
}

// Right ...
func (o *NN) Right(output []float64) []float64 {
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

	return o.Output
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

// Left ...
// func (o *NN) Left(data *StData) {
func (o *NN) Left(input, output []float64) {
	rdiff := make([]float64, len(o.Output))
	// 计算残差
	for k := range output {
		rdiff[k] = -(o.Output[k] - output[k]) * o.Output[k] * (1 - o.Output[k])
	}
	// o.ll.Log0Debug("残差:", rdiff)

	// 修正每层残差
	output1 := rdiff
	for index := len(o.Weight) - 1; index >= 0; index-- {
		// 输入层加权求和
		if index == 0 {
			output1 = o.matrixMul2(output1, index, input)
		} else {
			output1 = o.matrixMul2(output1, index, o.Hidden[index-1])
		}
	}

	// o.ll.Log0Debug("weight:", o.Weight)
}

// Init ...
func (o *NN) Init() error {
	o.ll = logger.NewLogger(nil)
	o.ll.SetFlags(log.Lshortfile)

	if o.MinDiff <= 0 {
		o.MinDiff = 0.0001
	}
	if o.Count <= 0 {
		o.Count = 1000
	}

	// fmt.Printf("name:%v | diff:%f | data: %v | count:%v | layer:%v\n", o.Name, o.MinDiff, len(o.Data), o.Count, o.Layer)

	// generate hidden layer
	for _, v := range o.Layer {
		o.Hidden = append(o.Hidden, make([]float64, v))
	}
	// o.ll.Log0Debug("hidden:", o.Hidden)

	// generate weight
	if o.Weight == nil {
		o.ResetWeight()
	}

	// log.Println(o.Weight)
	// log.Println(o.Bias)
	// os.Exit(0)

	// o.ll.Log4Trace("weight:", fmt.Sprintf("%0.2v", o.Weight))

	// 数据归一
	// if err := o.normalizing(&o.Data); err != nil {
	// 	return err
	// }
	// if err := o.normalizing(&o.Test); err != nil {
	// 	return err
	// }
	return nil
}

// ResetWeight ...
func (o *NN) ResetWeight() {
	o.Weight = make([][][]float64, 0)
	// generate weight
	tmp := []int{o.InputNum}
	tmp = append(tmp, o.Layer...)
	tmp = append(tmp, o.OutputNum)
	for i := 0; i < len(tmp)-1; i++ {
		t1 := make([][]float64, tmp[i])
		for j := 0; j < tmp[i]; j++ {
			t1[j] = make([]float64, tmp[i+1])
			for m := 0; m < len(t1[j]); m++ {
				if o.randFloat64(0.1, 0.9) < 0.5 {
					t1[j][m] = o.randFloat64(0.1, 0.9)
				} else {
					t1[j][m] = -o.randFloat64(0.1, 0.9)
				}
			}
		}
		o.Weight = append(o.Weight, t1)
	}
}

// 数据归一
func (o *NN) normalizing(data *[]StData) error {
	// k=（b-a)/(Max-Min)
	// Y=a+k(X-Min) 或者 Y=b+k(X-Max)
	// X=(Y-a)/k+Min

	// a/b/min/max/k
	a, b := 0.0, 1.0
	{ // 查找input最大最小值
		min, max := (*data)[0].input[0]*2, (*data)[0].input[0]/2
		for _, v := range *data {
			for _, vv := range v.input {
				if vv > max {
					max = vv
				}
				if vv < min {
					min = vv
				}
			}
		}
		if max == min {
			return errors.New("min == max")
		}
		if min < a || max > b {
			_k := (b - a) / (max - min)
			for k := range *data {
				for kk := range (*data)[k].input {
					(*data)[k].input[kk] = a + _k*((*data)[k].input[kk]-min)
				}
			}
		}
	}

	{ // 查找output最大最小值
		min, max := (*data)[0].input[0]*2, (*data)[0].input[0]/2
		for _, v := range *data {
			for _, vv := range v.output {
				if vv > max {
					max = vv
				}
				if vv < min {
					min = vv
				}
			}
		}
		if max == min {
			return errors.New("min == max")
		}
		if min < a || max > b {
			_k := (b - a) / (max - min)
			for k := range *data {
				for kk := range (*data)[k].output {
					(*data)[k].output[kk] = a + _k*((*data)[k].output[kk]-min)
				}
			}
		}
	}

	return nil
}

// Train ...
func (o *NN) Train() error {
	if err := o.Init(); err != nil {
		return err
	}

	all := o.Count * len(o.Data)
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

			if o.StudyCountCallback != nil {
				o.StudyCountCallback(study)
			}
			if study%(all/1000) == 0 {
				percent := o.Check(false, false)
				fmt.Printf("\r训练：%v/%v(%.1f%%) | 误差：%0.8f | 成功率：%.2f%%", study, all, float64(study)/float64(all)*100, max, percent*100)
			}
			if study%(all/10) == 0 {
				fmt.Println()
			}
		}

	}
	fmt.Println()
	fmt.Println("学习次数:", study)

	return nil
}

func (o *NN) train(v *StData, diff *[]float64) {
	runtime.Gosched()
	o.Right(v.input)
	o.Left(v.input, v.output)
	for kk := range v.output {
		if o.TestCallback != nil {
			(*diff)[kk] = o.TestCallback(o.Output[kk], v.output[kk])
		} else {
			(*diff)[kk] = o.defaultCheckFun(o.Output[kk], v.output[kk])
		}
	}
}

// Check ...
func (o *NN) Check(showLog bool, showPercent bool) float64 {
	chk := 0
	success := 0
	b := 0.0
	for _, v := range o.Test {
		chk++
		o.Right(v.input)
		if o.TestCallback != nil {
			b = o.TestCallback(v.output[0], o.Output[0])
		} else {
			b = o.defaultCheckFun(v.output[0], o.Output[0])
		}

		if b < o.MinDiff {
			success++
		}
		if showLog {
			fmt.Printf("\r检测:%v | 期望：%0.8f | 结果：%0.8f | 误差：%0.8f   ", b < o.MinDiff, v.output[0], o.Output[0], b)
		}
	}
	percent := float64(success) / float64(chk)
	if showPercent {
		fmt.Printf("\n检测:%v | 成功：%v | 成功率:%0.2f%%\n", chk, success, percent*100)
	}
	return percent
}

func (o NN) defaultCheckFun(chk, result float64) float64 {
	return math.Pow(chk-result, 2)
}

// mrand "math/rand"
func (o *NN) randFloat64(min, max float64) float64 {
	if min == 0 && max == 0 {
		return 0
	}
	if o.Seed {
		o.Seed = false
		mrand.Seed(time.Now().UnixNano())
	}
	for {
		x := mrand.Float64()*(max-min) + min
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

// SaveWeight ...
func (o *NN) SaveWeight(fileName string) error {
	bs, err := json.Marshal(o.Weight)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(fileName, bs, 0644)
}

// ToJSON ...
func (o *NN) ToJSON() string {
	bs, _ := json.Marshal(o.Weight)
	return string(bs)
}

// FromJSON ...
func (o *NN) FromJSON(str string) error {
	return json.Unmarshal([]byte(str), &o.Weight)
}
