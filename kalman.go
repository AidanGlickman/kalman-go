package kalman

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// KalmanFilter implementation targetting 2d constant velocity motion.
// Assumes accurate measurements of the position of the object.
// Assumes constant velocity.

const (
	// State vector size
	_N = 4
	// Measurement vector size
	_M = 2

	// State vector indices
	_X  = 0
	_Y  = 1
	_VX = 2
	_VY = 3

	// Initial uncertainty values
	// Position uncertainty is low because we assume accurate measurements
	_X_UNCERTAINTY = 1
	_Y_UNCERTAINTY = 1
	// Velocity uncertainty is high because we have no idea of the actual velocity
	_VX_UNCERTAINTY = 100
	_VY_UNCERTAINTY = 100

	// Measurement uncertainty values
	_M_UNCERTAINTY = 1
)

type KalmanFilter struct {
	state           *mat.VecDense // State vector
	stateTransition *mat.Dense    // State transition matrix
	dt              float64       // Time step
	cov             *mat.Dense    // Covariance matrix
	noise           *mat.SymDense // Noise matrix
	obs             *mat.Dense    // Observation matrix
	measCov         *mat.Dense    // Measurement covariance matrix
}

type StateWrapper struct {
	x  float64
	y  float64
	vx float64
	vy float64
}

func NewStateWrapper(x, y, vx, vy float64) *StateWrapper {
	return &StateWrapper{
		x:  x,
		y:  y,
		vx: vx,
		vy: vy,
	}
}

// Initialize the covariance matrix
// We initialize it based on the uncertainty of the initial state
// These values are described in the constants above
func InitUncertainty() *mat.Dense {
	cov := mat.NewDense(_N, _N, nil)
	cov.Set(_X, _X, _X_UNCERTAINTY)
	cov.Set(_Y, _Y, _Y_UNCERTAINTY)
	cov.Set(_VX, _VX, _VX_UNCERTAINTY)
	cov.Set(_VY, _VY, _VY_UNCERTAINTY)
	return cov
}

// Even with a constant velocity model, we can express the position
// and velocity covariances in terms of random acceleration variance.
// Some background on the probability of random acceleration:
// https://www.kalmanfilter.net/background2.html#exp
// Here we use a discrete noise model
func InitProcNoise(accelVar, dt float64) *mat.SymDense {
	noise := mat.NewSymDense(_N, nil)
	noise.SetSym(_X, _X, math.Pow(dt, 4)/2)
	noise.SetSym(_Y, _Y, math.Pow(dt, 4)/2)
	noise.SetSym(_VX, _VX, math.Pow(dt, 2))
	noise.SetSym(_VY, _VY, math.Pow(dt, 2))
	noise.SetSym(_X, _VX, math.Pow(dt, 3)/2)
	noise.SetSym(_Y, _VY, math.Pow(dt, 3)/2)

	noise.ScaleSym(accelVar, noise)

	return noise
}

func InitStateTransition(dt float64) *mat.Dense {
	stateTransition := mat.NewDense(_N, _N, nil)
	// Start with identity matrix
	for i := 0; i < _N; i++ {
		stateTransition.Set(i, i, 1)
	}
	// Change in position is dependent on time step and velocity
	stateTransition.Set(_X, _VX, dt)
	stateTransition.Set(_Y, _VY, dt)

	return stateTransition
}

// The observation Matrix acts as a method of extracting the observable
// state from the state vector.
func InitObservationMatrix() *mat.Dense {
	observationMatrix := mat.NewDense(_M, _N, nil)
	observationMatrix.Set(_X, _X, 1)
	observationMatrix.Set(_Y, _Y, 1)
	return observationMatrix
}

// The measurement uncertainty accounts for random variation in the
// accuracy of our measurements. This model uses one value, although accuracy
// may be improved by factoring in things like SNR and camera specs.
// This function also assumes no correlation between the measurements for x and y.
func InitMeasurementUncertainty() *mat.Dense {
	measurementUncertainty := mat.NewDense(_M, _M, nil)
	measurementUncertainty.Set(_X, _X, _M_UNCERTAINTY)
	measurementUncertainty.Set(_Y, _Y, _M_UNCERTAINTY)

	return measurementUncertainty
}

func NewKalmanFilter(state StateWrapper, dt float64) *KalmanFilter {
	return &KalmanFilter{
		state:           mat.NewVecDense(_N, []float64{state.x, state.y, state.vx, state.vy}),
		stateTransition: InitStateTransition(dt),
		dt:              dt,
		cov:             InitUncertainty(),
		noise:           InitProcNoise(0.1, dt),
		obs:             InitObservationMatrix(),
		measCov:         InitMeasurementUncertainty(),
	}
}

func (kf *KalmanFilter) PredictState() *mat.VecDense {
	predState := mat.NewVecDense(_N, nil)
	predState.MulVec(kf.stateTransition, kf.state)

	return predState
}

func (kf *KalmanFilter) PredictCov() *mat.Dense {
	predCov := mat.NewDense(_N, _N, nil)
	predCov.Mul(kf.stateTransition, kf.cov)
	predCov.Mul(kf.cov, kf.stateTransition.T())
	predCov.Add(kf.cov, kf.noise)

	return predCov
}

func (kf *KalmanFilter) KalmanGain() *mat.Dense {
	PriorMeasCov := mat.NewDense(_M, _N, nil)
	PriorMeasCov.Mul(kf.obs, kf.cov)
	PriorMeasCov.Mul(PriorMeasCov, kf.obs.T())
	PriorMeasCov.Add(PriorMeasCov, kf.measCov)
	PriorMeasCov.Inverse(PriorMeasCov)

	kGain := mat.NewDense(_N, _M, nil)
	kGain.Mul(kf.cov, kf.obs.T())
	kGain.Mul(kGain, PriorMeasCov)

	return kGain
}

func (kf *KalmanFilter) UpdateState(meas *mat.VecDense) {
	kGain := kf.KalmanGain()

	predictedMeas := mat.NewVecDense(_M, nil)
	predictedMeas.MulVec(kf.obs, kf.state)

	err := mat.NewVecDense(_M, nil)
	err.SubVec(meas, predictedMeas)

	kChange := mat.NewVecDense(_N, nil)
	kChange.MulVec(kGain, err)
	kf.state.AddVec(kf.state, kChange)
}

func (kf *KalmanFilter) UpdateCov(meas *mat.VecDense) {

	kGain := kf.KalmanGain()
	I := MakeIdentityMatrix(_N)

	obsGain := mat.NewDense(_N, _M, nil)
	obsGain.Mul(kGain, kf.obs)

	obsDiff := mat.NewDense(_N, _N, nil)
	obsDiff.Sub(I, obsGain)

	kf.cov.Mul(obsDiff, kf.cov)
	kf.cov.Mul(kf.cov, obsDiff.T())

	uncGain := mat.NewDense(_N, _N, nil)
	uncGain.Mul(kGain, kf.noise)
	uncGain.Mul(uncGain, kGain.T())

	kf.cov.Add(kf.cov, uncGain)
}

func (kf *KalmanFilter) Update(meas *mat.VecDense) {
	kf.UpdateState(meas)
	kf.UpdateCov(meas)
}

func (kf *KalmanFilter) Predict() (predState *mat.VecDense, predCov *mat.Dense) {
	predState = kf.PredictState()
	predCov = kf.PredictCov()

	return predState, predCov
}
