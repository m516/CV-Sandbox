#pragma once
#include <initializer_list>
#include <limits>
#include <string>


namespace Soft {

	template<
		typename T, //real type
		typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
	> // Force numeric
		class Property {
		public:
			Property() : value(0), target(0), ownsData(false) {};
			Property(const Property &other) : ownsData(other.ownsData) {
				if (ownsData) {
					value = new T(*other.value);
					target = new T(*other.target);
				}
				else {
					value = other.value;
					target = other.target;
				}
			};
			Property(T initialValue) : value(initialValue), target(initialValue), ownsData(true) {};
			Property(T v, T t) : value(new T(v)), target(new T(t)), ownsData(true) {};
			Property(T* v, T* t) : value(v), target(t), ownsData(false) {};
			virtual ~Property() {
				if (ownsData) {
					delete target;
					delete value;
				}
			}
			T* target;
			T* value;
			T force(T newValue) {
				// Make sure integers don't get stuck
				if (newValue == *value) *value = *target;
				else *value = newValue;
				return *value;
			}
			T stepLinear(T howMuch) {
				if (*target - *value > howMuch) *value += howMuch;
				else if (*target - *value < -howMuch) *value -= howMuch;
				else *value = *target;
			}
			T stepExp(float timestep) {
				float newValue = (*target * timestep) + (*value * (1.f - timestep));
				return force((T)newValue);
			}
			T stepExp(double timestep) {
				double newValue = (*target * timestep) + (*value * (1.0 - timestep));
				return force((T)newValue);
			}
			operator T* () { return target; }
			operator T() { return *value; }
			Property& operator= (const Property<T> &other) { 
				// Guard self assignment
				if (this == &other)
					return *this;
				if (ownsData) {
					delete target;
					delete value;
				}
				ownsData = other.ownsData;
				if (ownsData) {
					value = new T(*other.value);
					target = new T(*other.target);
				}
				else {
					value = other.value;
					target = other.target;
				}
				return *this;
			}
		private:
			bool ownsData;
	};

	template<
		std::size_t D,
		typename T, //real type
		typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
	> // Force numeric
		class PropertyArray {
		public:
			std::vector<T> targets;
			std::vector<T> values;
			PropertyArray() : values(std::vector<T>(0)), targets(std::vector<T>(0)) {
				for (int i = 0; i < D; i++)
					properties[D] = Property<T>(&values[i], &targets[i]);
			}
			PropertyArray(std::initializer_list<T> initialValues) : values(initialValues), targets(initialValues){
				for (int i = 0; i < D; i++) {
					properties[i] = Property<T>(values.data()+i, targets.data()+i);
				}
			}
			void force(T* newValue) {
				for (int i = 0; i < D; i++) properties[i].force(newValue[i]);
			}
			void stepLinear(T howMuch) {
				for (int i = 0; i < D; i++) properties[i].stepLinear(howMuch);
			}
			void stepLinear(T *howMuches) {
				for (int i = 0; i < D; i++) properties[i].stepLinear(howMuches[i]);
			}
			void stepExp(float timestep) {
				for (int i = 0; i < D; i++) properties[i].stepExp(timestep);
			}
			void stepExp(float *timesteps) {
				for (int i = 0; i < D; i++) properties[i].stepExp(timesteps[i]);
			}
			void stepExp(double timestep) {
				for (int i = 0; i < D; i++) properties[i].stepExp(timestep);
			}
			void stepExp(double *timesteps) {
				for (int i = 0; i < D; i++) properties[i].stepExp(timesteps[i]);
			}
			operator T* () { return values.data(); }
	private:
		Property<T> properties[D];
	};

}