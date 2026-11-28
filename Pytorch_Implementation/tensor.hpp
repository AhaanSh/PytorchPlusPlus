#pragma once

#include <vector>
#include <memory> 
#include <cstdint> 
#include <stdexcept> 
#include <utility> 
#include <algorithm> 
#include "storage.hpp"

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

    //walks to the next index and returns true if it is valid, also modifies index by reference 
    static bool walkIdx(std::vector<int64_t>& index, const std::vector<int64_t>& shapeIn) {
        for (int64_t i = static_cast<int64_t>(shapeIn.size()) - 1; i >= 0; i--) {
            if (++index[i] < shapeIn[i]) return true; 
            index[i] = 0; 
        }

        return false; 
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
    //TO-DO: Can template this function so we don't need to allocate a vector each time 
    const float& operator()(const std::vector<int64_t>& indices) const {
         return (*storage)[flatIdx(indices)];  
    }

    //operator() overload -- returns a reference to a specific element in the Tensor, allowing rewriting
    float& operator()(const std::vector<int64_t>& indices) {
        return (*storage)[flatIdx(indices)]; 
    }

    //squeeze -- removes a size 1 dimension and returns a new Tensor 
    Tensor squeeze(int64_t dimension) const {
        if (dimension < 0 || dimension >= dim()) throw std::out_of_range("Dimension out of range"); 
        if (shape[dimension] != 1) throw std::invalid_argument("Dimension expected size 1"); 

        //compute new shape without dim
        std::vector<int64_t> newShape; 
        std::vector<int64_t> newStrides; 
        for (int64_t i = 0; i < dim(); i++) {
            if (i != dimension) {
                newShape.push_back(shape[i]); 
                newStrides.push_back(strides[i]); 
            }
        }

        return Tensor(storage, std::move(newShape), std::move(newStrides), offset); 
    }

    //unsqueeze -- adds a size 1 dimension 
    Tensor unsqueeze(int64_t dimension) const {
        if (dimension < 0 || dimension > dim()) throw std::out_of_range("Dimension out of range"); 

        //compute new shape 
        std::vector<int64_t> newShape = shape; 
        std::vector<int64_t> newStrides = strides; 
        newShape.insert(newShape.begin() + dimension, 1); 
        newStrides.insert(newStrides.begin() + dimension, 1); 

        return Tensor(storage, std::move(newShape), std::move(newStrides), offset); 
    }

    //transpose -- computes the default transpose (reverse all axes) and returns new Tensor 
    Tensor transpose() const {
        //need to reverse shapes and strides vectors 
        std::vector<int64_t> newShape = shape; 
        std::vector<int64_t> newStrides = strides; 
        std::reverse(newShape.begin(), newShape.end()); 
        std::reverse(newStrides.begin(), newStrides.end()); 
        return Tensor(storage, std::move(newShape), std::move(newStrides), offset); 
    }

    //transpose -- swaps dim0 and dim1
    Tensor transpose(int64_t dim0, int64_t dim1) const {
        if (dim0 < 0 || dim0 >= dim() || dim1 < 0 || dim1 >= dim()) throw std::out_of_range("Dimension out of range"); 
        std::vector<int64_t> newShape = shape; 
        std::vector<int64_t> newStrides = strides; 
        std::swap(newShape[dim0], newShape[dim1]); 
        std::swap(newStrides[dim0], newStrides[dim1]); 
        return Tensor(storage, std::move(newShape), std::move(newStrides), offset); 
    }

    //isContigious -- checks if a tensor matches default row-major strides for its shape 
    bool isContiguous() const {
        int64_t expected = 1; 
        for (int64_t i = dim() - 1; i >= 0; i--) {
            if (shape[i] == 1) continue; 
            if (strides[i] == expected) {
                expected *= shape[i]; 
            } else return false; 
        }

        return true; 
    }

    //contigious -- returns a contigious version of the tensor 
    Tensor contiguous() const {
        if (isContiguous() || numel() == 0) return *this; 

        //create new storage -- fill with 0's for now 
        std::shared_ptr<Storage> newStorage = std::make_shared<Storage>(numel(), 0.0f); 

        //visit all elements of *this in row-major order and place them in storage 
        std::vector<int64_t> idx(dim(), 0); 
        bool walk = true; 
        int64_t i = 0; 
        while (walk) {
            (*newStorage)[i]= (*this)(idx); 
            walk = walkIdx(idx, shape); 
            i++; 
        }
        
        //return new tensor 
        return Tensor(std::move(newStorage), shape, defaultStrides(shape), 0); 
    }
}; 


