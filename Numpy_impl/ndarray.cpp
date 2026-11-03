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

class Tensor {
    std::shared_ptr<Storage> storage; 
    std::vector<int64_t> shape; 
    std::vector<int64_t> strides; 
    int64_t offset; 

private: 

    //return the default strides vector given the shape 
    static std::vector<int64_t> defaultStrides(const std::vector<int64_t>& shapes) {
        int64_t total = 1; 
        std::vector<int64_t> tempStrides(shapes.size(), 0); 
        for (int64_t i = static_cast<int64_t>(shapes.size()) - 1; i >= 0; i--) {
            tempStrides[i] = total; 
            total *= shapes[i]; 
        }

        return tempStrides; 
    }

    //finds the product of all the elements in shapes to return the total number of elements 
    static int64_t numel(const std::vector<int64_t>& shapes) {
        int64_t total = 1; 
        for (int64_t s : shapes) total *= s; 
        return total; 
    }

    int64_t flatIdx(const std::vector<int64_t>& indices) const {
        if (static_cast<int64_t>(indices.size()) != dim()) throw std::invalid_argument("Number of indices does not match required amount"); 
        //formula to calculate flattened index: Slot = offset + i0s0 + i1s1 + ... + insn
        int64_t res = offset; 
        for (int64_t i = 0; i < dim(); i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) throw std::out_of_range("Index out of bounds"); 
            res += (indices[i] * strides[i]); 
        }
        
        return res; 
    }


public: 

    explicit Tensor(std::shared_ptr<Storage> s, std::vector<int64_t> shape_in, std::vector<int64_t> strides_in, int64_t offset_in) : storage(std::move(s)), shape(std::move(shape_in)), strides(std::move(strides_in)), offset(offset_in) {
        if (shape.size() != strides.size()) throw std::invalid_argument("Shape and Strides need same size"); 
        if (offset < 0) throw std::invalid_argument("Can't have negative offset"); 
        if (storage == nullptr) throw std::invalid_argument("Can't have null storage"); 
        for (int64_t elem : shape) {
            if (elem < 0) throw std::invalid_argument("Shape can't have negative argument"); 
        }
    }


    //return a tensor of specific shape and fill with val
    static Tensor full(const std::vector<int64_t>& shapes, float val) {
        std::vector<int64_t> stridesIn = defaultStrides(shapes); 
        int64_t numElem = numel(shapes); 
        std::shared_ptr<Storage> s = std::make_shared<Storage>(numElem, val); 
        return Tensor(std::move(s), shapes, std::move(stridesIn), 0); 
    }

    //return Tensor of shape filled with 0's 
    static Tensor zeros(const std::vector<int64_t>& shapes) {
        return full(shapes, 0.0f); 
    }

    //return Tensor of Shape filled with 1's 
    static Tensor ones(const std::vector<int64_t>& shapes) {
        return full(shapes, 1.0f); 
    }

    //returns a tensor with offset 0 with data from data 
    static Tensor fromData(std::vector<float> data, std::vector<int64_t> shape_in) {
        if (numel(shape_in) != static_cast<int64_t>(data.size())) throw std::invalid_argument("Shape does not match data size"); 
        std::vector<int64_t> strides_in = defaultStrides(shape_in); 
        std::shared_ptr<Storage> s = std::make_shared<Storage>(std::move(data)); 
        return Tensor(std::move(s), std::move(shape_in), std::move(strides_in), 0); 
    }

    //accessor functions 

    //returns a const reference to shape for a Tensor 
    const std::vector<int64_t>& getShape() const {
        return shape; 
    }

    //returns a const reference to strides for a Tensor
    const std::vector<int64_t>& getStrides() const {
        return strides; 
    }

    //returns offset value for a Tensor 
    int64_t getOffset() const {
        return offset; 
    }

    //returns dimension of shape for a Tensor
    int64_t dim() const {
        return static_cast<int64_t>(shape.size()); 
    }

    //returns total number of elements in a Tensor 
    int64_t numel() const {
        return numel(shape); 
    }

    //operator() overload -- returns a specific element in the Tensor with a multi-dimensional index 
    const float& operator()(const std::vector<int64_t>& indices) const {
         return (*storage)[flatIdx(indices)];  
    }

    //operator() overload -- returns a reference to a specific element in the Tensor, allowing rewriting
    float& operator()(const std::vector<int64_t>& indices) {
        return (*storage)[flatIdx(indices)]; 
    }
}; 


