package main

import (
	"fmt"
	"math"
	mrand "math/rand"
	"net/http"
	"nn"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"syscall"
	"time"
)

// ...
const (
	WIDTH    = 20
	HEIGHT   = 10
	SIGEMPTY = 0 // empty
	SIGWALL  = 1 // wall
	SIGBLOCK = 2 // block
	SIGCAR   = 3 // car
)

var (
	screen   [][]int
	alive    = 0
	score    = 0
	maxScore = 0
	carCount = 10
	topUnit  = 4
	gen      = 0
	cars     = []*CAR{}
	ais      []*nn.NN
	speed    = 1000
	speedWeb = 50
	dis      = 0
)

func main() {

	http.HandleFunc("/speed", func(w http.ResponseWriter, r *http.Request) {
		s := r.FormValue("speed")
		if s == "" {
			return
		}
		speedWeb, _ = strconv.Atoi(s)
	})
	http.HandleFunc("/block", func(w http.ResponseWriter, r *http.Request) {
		s := r.FormValue("block")
		if s == "/" {
			screen[0][cars[0].x] = SIGBLOCK
			screen[1][cars[0].x-1] = SIGBLOCK
		}
		if s == "\\" {
			screen[0][cars[0].x] = SIGBLOCK
			screen[1][cars[0].x+1] = SIGBLOCK
		}
	})
	go http.ListenAndServe(":80", nil)

	for i := 0; i < carCount; i++ {
		cars = append(cars, newCar())
		ais = append(ais, &nn.NN{InputNum: WIDTH + 2, OutputNum: 1, Layer: []int{3}})
		ais[i].Init()
	}

	restart()

	go func() {
		for {
			update()
			time.Sleep(time.Second / time.Duration(speed))
		}
	}()

	c := make(chan os.Signal)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
	<-c
}

func restart() {
	score = 0
	screen = make([][]int, HEIGHT)
	for i := 0; i < HEIGHT; i++ {
		screen[i] = make([]int, WIDTH)
		screen[i][0] = SIGWALL
		screen[i][WIDTH-1] = SIGWALL
	}

	for i := 0; i < carCount; i++ {
		cars[i].restart()
	}
}

func update() {
	newLine := make([]int, WIDTH)
	newLine[0], newLine[WIDTH-1] = SIGWALL, SIGWALL

	dis--
	if dis <= 0 {
		dis = WIDTH/3 + 2
		w := WIDTH / 3
		x := RandIntn(1, WIDTH-1-w)
		for i := x; i < x+w; i++ {
			if i < 1 || i > WIDTH-2 {
				continue
			}
			newLine[i] = SIGBLOCK
		}
	}
	screen = append([][]int{newLine}, screen[:HEIGHT-1]...)

	draw()

	score++
	if score > maxScore {
		maxScore = score
	}
}

func draw() {
	out := ""
	out += "\033c"

	// car
	alive = 0
	for index, car := range cars {

		inputs := car.getInputs()
		res := ais[index].Right(inputs)
		// fmt.Println(inputs, res)
		cars[index].control(int(math.Round(res[0] * WIDTH)))

		cars[index].update()
		if !car.alive {
			continue
		}
		alive++
		screen[HEIGHT-1][car.x] = SIGCAR
	}

	if alive == 0 {
		speed = 10000
		next()
		restart()
	}
	if score > 5000 {
		speed = speedWeb
	}

	for _, y := range screen {
		for _, x := range y {
			if x == SIGWALL {
				out += "|"
			} else if x == SIGBLOCK {
				out += "o"
			} else if x == SIGCAR {
				out += "A"
			} else {
				out += " "
			}
		}
		out += "\n"
	}
	out += fmt.Sprintf("gen: %d\n", gen)
	out += fmt.Sprintf("score: %d/%d\n", score, maxScore)
	out += fmt.Sprintf("alive: %d/%d\n", alive, len(cars))
	fmt.Print(out)
}

func next() {
	st := NNSlice{}
	for i := 0; i < carCount; i++ {
		st = append(st, NNSliceSub{Index: i, Score: cars[i].maxScore})
	}
	sort.Sort(sort.Reverse(st))

	gen++

	for i := topUnit; i < carCount; i++ {
		if i == 4 {
			copyWeightBias(ais[i], cross(ais[st[0].Index], ais[st[1].Index]))
		} else if i < carCount-2 {
			copyWeightBias(ais[i], cross(ais[st[RandIntn(0, 3)].Index], ais[st[RandIntn(0, 3)].Index]))
		} else {
			copyWeightBias(ais[i], ais[st[RandIntn(0, 3)].Index])
		}
		mutation(ais[i])
	}
}

func copyWeightBias(dst, src *nn.NN) {
	for k1, v1 := range src.Weight {
		for k2, v2 := range v1 {
			for k3 := range v2 {
				dst.Weight[k1][k2][k3] = src.Weight[k1][k2][k3]
			}
		}
	}
}

func cross(a, b *nn.NN) *nn.NN {
	m := &nn.NN{InputNum: a.InputNum, OutputNum: a.OutputNum, Layer: a.Layer}
	n := &nn.NN{InputNum: b.InputNum, OutputNum: b.OutputNum, Layer: b.Layer}
	m.Init()
	n.Init()
	copyWeightBias(m, a)
	copyWeightBias(n, b)
	cutPoint := RandIntn(0, len(a.Weight))
	for i := cutPoint; i < len(a.Weight); i++ {
		for k1, v1 := range a.Weight[i] {
			for k2 := range v1 {
				x := a.Weight[i][k1][k2]
				a.Weight[i][k1][k2] = b.Weight[i][k1][k2]
				b.Weight[i][k1][k2] = x
			}
		}
	}
	return map[int]*nn.NN{0: m, 1: n}[RandIntn(0, 1)]
}

func mutate(gene float64) float64 {
	if RandIntn(1, 10) <= 2 {
		gene += gene*(mrand.Float64()-0.5)*3 + (mrand.Float64() - 0.5)
	}
	return gene
}
func mutation(x *nn.NN) {
	for k1, v1 := range x.Weight {
		for k2, v2 := range v1 {
			for k3 := range v2 {
				x.Weight[k1][k2][k3] = mutate(x.Weight[k1][k2][k3])
			}
		}
	}
}

// RandIntn return min <= x <= max
// mrand "math/rand"
func RandIntn(min, max int) int {
	if min == 0 && max == 0 {
		return 0
	}
	mrand.Seed(time.Now().UnixNano())
	return mrand.Intn(max+1-min) + min
	// return mrand.New(mrand.NewSource(time.Now().UnixNano())).Intn((max-min)+1) + min
}

type NNSliceSub struct {
	Index int
	Score int
}
type NNSlice []NNSliceSub

func (c NNSlice) Len() int {
	return len(c)
}
func (c NNSlice) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}
func (c NNSlice) Less(i, j int) bool {
	return c[i].Score < c[j].Score
}
