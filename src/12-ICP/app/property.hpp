#pragma once
#include <limits>
#include <string>


namespace Soft {

	template<
		typename T, //real type
		typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
	> // Force numeric
	class Property {
	public:
		Property(T initialValue) :value(initialValue), target(initialValue) {};
		Property(T v, T t) :value(v), target(t) {};
		T target;
		T value;
		T force(T newValue) {
			// Make sure integers don't get stuck
			if (newValue == value) value = target;
			else value = newValue;
			return value;
		}
		T stepLinear(T howMuch) {
			if (target - value > howMuch) value += howMuch;
			else if (target - value < -howMuch) value -= howMuch;
			else value = target;
		}
		T stepExp(float timestep) {
			float newValue = (target * timestep) + (value * (1.f - timestep));
			return force((T)newValue);
		}
		T stepExp(double timestep) {
			double newValue = (target * timestep) + (value * (1.0 - timestep));
			return force((T)newValue);
		}
		operator T*() { return &target; }
		operator T() { return value; }
		
	};

}