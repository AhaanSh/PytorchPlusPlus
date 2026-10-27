#include <memory> 
#include <vector> 
#include <cstdint>
#include <stdexcept> 
#include <string> 
#include <utility> 

class Storage {
    //flat buffer to store data 
    std::vector<float> mem; 

public: 
    //constructor that moves a flattened 1D vector into mem 
    explicit Storage(std::vector<float> flattened) : mem(std::move(flattened)) {}

    //constructor that takes a number of elements and fills with a specific value 
    explicit Storage(int64_t count, float val) {
        if (count < 0) throw std::invalid_argument("negative count"); 
        mem = std::vector<float>(static_cast<size_t>(count), val); 
    }

    //return the number of elements in mem 
    int64_t size() const {
        return static_cast<int64_t>(mem.size()); 
    }

    //return an element at a flattened index
    const float& operator[](int64_t i) const {
        if (i < 0 || i >= size()) throw std::out_of_range("Index out of bounds"); 
        return mem[i]; 
    }

    //return a reference to an element in mem allowing for writing to index 
    float& operator[](int64_t i) {
        if (i < 0 || i >= size()) throw std::out_of_range("Index out of bounds"); 
        return mem[i]; 
    }


}; 