#pragma once
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

class MinHeapWithTracking {
public:
    MinHeapWithTracking() = default;

    void push(const std::pair<int, int>& item);

    std::pair<int, int> pop();

    std::pair<int, int> peek() const;

    bool isEmpty() const;

    void incrementCount(int id, int add_count);

    friend std::ostream& operator<<(std::ostream& os, const MinHeapWithTracking& heap);

private:
    std::vector<std::pair<int, int>> heap;
    std::unordered_map<int, int> id_to_index;

    void heapify(int index);

    void siftUp(int index);

    void swap(int i, int j);
};
