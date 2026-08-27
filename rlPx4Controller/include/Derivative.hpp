#pragma once
#include <cmath>
class FilterBase
{
public:
    /**
     * @brief Default Virtual Functions
     * @param input Input Value
     * @param dt Delta t Between Two Observation Loop
     */
    virtual float update(float input, float dt) = 0; 

    void setFCut(float fCut) { this->fCut_ = fCut; }
    void setState(float state) { state_ = state; }

    float getDt() { return Dt_; }
    float getFCut() { return fCut_; }
    float getState() { return state_; }

protected:
    float state_{NAN}; // initialize to invalid val, force into is_finite() check on first call

    float fCut_;
    float Dt_;
};

class LowPassFilter : public FilterBase
{
public:
    /**
     * @brief Update Functions for LowPassFilter
     * @param input Input Value
     * @param dt Delta t Between Two Observation Loop
     */
    LowPassFilter() : FilterBase() {}

    float update(float input, float dt) override
    {
        if (std::isnan(getState()))
        {
            setState(input);
        }
        Dt_ = dt;
        float b = 2 * float(M_PI) * getFCut() * getDt();
        float a = b / (1 + b);
        state_ = a * input + (1 - a) * getState();
        return getState();
    }
};

class Derivate
{
public:
    Derivate() : initialized_(false) 
    {
        _lowPass.setFCut(10); //Default Cutoff Frequence
    }

    float update(float input, float dt)
    {
        setDt(dt);
        float output;

        if (initialized_)
        {
            // input = _lowPass.update(input, getDt());
            output = _lowPass.update((input - getU()) / getDt(), getDt());
        }
        else
        {
            // if this is the first call to update
            // we have no valid derivative
            // and so we use the assumption the
            // input value is not changing much,
            // which is the best we can do here.
            _lowPass.update(0.0f, getDt());
            output = 0.0f;
            initialized_ = true;
        }
        setU(input);
        return output;
    }
    // accessors

private:
    float Dt_;
    float _u; /**< previous input */
    bool initialized_;
    LowPassFilter _lowPass; /**< low pass filter */

    void setU(float u) { _u = u; }
    void reset() { initialized_ = false; };
    void setDt(float dt) { this->Dt_ = dt; };
    float getDt() { return Dt_; };
    float getU() { return _u; }
    float getLP() { return _lowPass.getFCut(); }
    float getO() { return _lowPass.getState(); }
};
