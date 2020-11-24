package main

// CAR ...
type CAR struct {
	alive    bool
	score    int
	maxScore int
	x        int
}

func newCar() *CAR {
	o := &CAR{}
	o.restart()
	return o
}

func (o *CAR) restart() {
	o.score = 0
	o.alive = true
	o.x = WIDTH / 2
}

func (o *CAR) getInputs() []float64 {
	blockIndex := 0
firstFor:
	for i := HEIGHT - 1; i >= 0; i-- {
		for j := 0; j < WIDTH; j++ {
			if screen[i][j] == SIGBLOCK {
				blockIndex = i
				break firstFor
			}
		}
	}

	block := make([]float64, WIDTH)
	for x := 0; x < WIDTH; x++ {
		if screen[blockIndex][x] == SIGBLOCK {
			block[x] = 1
		}
	}
	return append(block, float64(blockIndex)/float64(HEIGHT), float64(o.x)/float64(WIDTH))
}
func (o *CAR) control(x int) {

	if x < 1 {
		x = 1
	}
	if x > WIDTH-2 {
		x = WIDTH - 2
	}

	if x > o.x {
		o.x++
	} else if x < o.x {
		o.x--
	}
}
func (o *CAR) update() {
	o.chk()
	if !o.alive {
		return
	}

	o.score = score
	if o.score > o.maxScore {
		o.maxScore = o.score
	}
}

func (o *CAR) chk() {
	if screen[HEIGHT-1][o.x] == SIGBLOCK {
		o.alive = false
	}
}
