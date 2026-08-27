#ifndef __PX4_MIXER
#define __PX4_MIXER
#include <Eigen/Eigen>
// #include <pybind11/eigen.h>
#include "MyMath.hpp"
#include <cfloat>
struct Rotor
{
    float roll_scale;   /**< scales roll for this rotor */
    float pitch_scale;  /**< scales pitch for this rotor */
    float yaw_scale;    /**< scales yaw for this rotor */
    float thrust_scale; /**< scales thrust for this rotor */
};

static constexpr Rotor _config_quad_x[]{
        {-0.707107, -0.707107, -1.000000, 1.000000},
        {0.707107, 0.707107, -1.000000, 1.000000},
        {0.707107, -0.707107, 1.000000, 1.000000},
        {-0.707107, 0.707107, 1.000000, 1.000000},
        
};
class Px4Mixer
{
private:
    Eigen::Vector4d _tmp_array;
    uint8_t _rotor_count;
    Eigen::Vector4d _rotor_outputs;
    double _thrust_factor = 1;
    /* data */
public:
    Px4Mixer(/* args */);
    ~Px4Mixer();

    Eigen::Vector4d update(const Eigen::Vector4d  &torque);
    float compute_desaturation_gain(Eigen::Vector4d desaturation_vector, Eigen::Vector4d& outputs, float min_output, float max_output);

    // float compute_desaturation_gain(const float *desaturation_vector, const float *outputs, float min_output, float max_output);
    // void minimize_saturation(
    //     const float *desaturation_vector, // Values on the axes (pitch, roll, yaw) that need desaturation
    //     float *outputs,
    //     float min_output,
    //     float max_output,
    //     bool reduce_only);
    void minimize_saturation(Eigen::Vector4d desaturation_vector,// Values on the axes (pitch, roll, yaw) that need desaturation
        Eigen::Vector4d& outputs,
        float min_output,
        float max_output,
        bool reduce_only=false);
};

Px4Mixer::Px4Mixer(/* args */)
{
    _rotor_count = 4;
}

Px4Mixer::~Px4Mixer()
{
}
Eigen::Vector4d Px4Mixer::update(const Eigen::Vector4d  &torque)
{
    float roll = MyMath::constrain(torque(0), (double)-1.0f, (double)1.0f);
    float pitch = MyMath::constrain(torque(1), (double)-1.0f, (double)1.0f);
    float yaw = MyMath::constrain(torque(2), (double)-1.0f, (double)1.0f);
    float thrust = MyMath::constrain(torque(3), (double)0.0f, (double)1.0f);
    // Airmode disabled: never allow to increase the thrust to unsaturate a motor

    // Mix without yaw
    for (unsigned i = 0; i < _rotor_count; i++)
    {
        _rotor_outputs(i) = roll * _config_quad_x[i].roll_scale +
                     pitch * _config_quad_x[i].pitch_scale +
                     thrust * _config_quad_x[i].thrust_scale;
                    
        // Thrust will be used to unsaturate if needed
        _tmp_array[i] = _config_quad_x[i].thrust_scale;
    }
    //only reduce thrust
    minimize_saturation(_tmp_array,_rotor_outputs,0.f,1.f,true);

    // reduce roll 
    for (unsigned i = 0; i < _rotor_count; i++)
    {                    
        _tmp_array[i] = _config_quad_x[i].roll_scale;
    }
    minimize_saturation(_tmp_array,_rotor_outputs,0.f,1.f);

    // reduce pitch
    for (unsigned i = 0; i < _rotor_count; i++)
    {                    
        _tmp_array[i] = _config_quad_x[i].pitch_scale;
    }

    minimize_saturation(_tmp_array,_rotor_outputs,0.f,1.f);

    // reduce yaw 
    for (unsigned i = 0; i < _rotor_count; i++)
    {      
        // add yaw to outputs
        _rotor_outputs(i) += yaw * _config_quad_x[i].yaw_scale;

        _tmp_array[i] = _config_quad_x[i].yaw_scale;
    }
    minimize_saturation(_tmp_array,_rotor_outputs,0.f,1.15f);
    for (unsigned i = 0; i < _rotor_count; i++)
    {                    
        _tmp_array[i] = _config_quad_x[i].thrust_scale;
    }
    minimize_saturation(_tmp_array,_rotor_outputs,0.f,1.f,true);
    for (unsigned i = 0; i < _rotor_count; i++)
    {                    
        _rotor_outputs(i) = MyMath::constrain(_rotor_outputs(i), (double)0.0f, (double)1.0f);
    }

        // Apply thrust model and scale outputs to range [idle_speed, 1].
    // At this point the outputs are expected to be in [0, 1], but they can be outside, for example
    // if a roll command exceeds the motor band limit.
    for (unsigned i = 0; i < _rotor_count; i++) {
      // Implement simple model for static relationship between applied motor pwm and motor thrust
      // model: thrust = (1 - _thrust_factor) * PWM + _thrust_factor * PWM^2
    //   if (_thrust_factor > 0.0f) {
    //     _rotor_outputs[i] = -(1.0f - _thrust_factor) / (2.0f * _thrust_factor) + sqrtf((1.0f - _thrust_factor) *
    //         (1.0f - _thrust_factor) / (4.0f * _thrust_factor * _thrust_factor) + (_rotor_outputs[i] < 0.0f ? 0.0f : _rotor_outputs[i] /
    //             _thrust_factor));
    //     }
      }



    return _rotor_outputs;
}

float Px4Mixer::compute_desaturation_gain(Eigen::Vector4d desaturation_vector, Eigen::Vector4d& outputs, float min_output, float max_output)
{
    float k_min = 0.f;
    float k_max = 0.f;
    for (unsigned i = 0; i < _rotor_count; i++)
    {
        // Avoid division by zero. If desaturation_vector[i] is zero, there's nothing we can do to unsaturate anyway
        if (fabsf(desaturation_vector[i]) < FLT_EPSILON)
        {
            continue;
        }
        if (outputs[i] < min_output) 
        {
            float k = (min_output - outputs[i]) / desaturation_vector[i];
            if (k < k_min)
            {
                k_min = k;
            }
            if (k > k_max)
            {
                k_max = k;
            }
        }
        if (outputs[i] > max_output)
        {
            float k = (max_output - outputs[i]) / desaturation_vector[i];
            if (k < k_min)
            {
                k_min = k;
            }
            if (k > k_max)
            {
                k_max = k;
            }
        }
    }
    // Reduce the saturation as much as possible
    return k_min + k_max;
}

void Px4Mixer::minimize_saturation(
    Eigen::Vector4d desaturation_vector, 
    Eigen::Vector4d &outputs,
    float min_output,
    float max_output,
    bool reduce_only)
{
    float k1 = compute_desaturation_gain(desaturation_vector, outputs, min_output, max_output);

    // When disabled, reduce_only=true, meaning k1 will only reduce if it is negative.
    if (reduce_only && k1 > 0.f)
    {
        return;
    }
    for (unsigned i = 0; i < _rotor_count; i++)
    {
        outputs[i] += k1 * desaturation_vector[i];
    }

    // Compute the desaturation gain again based on the updated outputs.
    // In most cases it will be zero. It won't be if max(outputs) - min(outputs) > max_output - min_output.
    // In that case adding 0.5 of the gain will equilibrate saturations.

    float k2 = 0.5f * compute_desaturation_gain(desaturation_vector, outputs, min_output, max_output);

    for (unsigned i = 0; i < _rotor_count; i++)
    {
        outputs[i] += k2 * desaturation_vector[i];
    }
}
#endif