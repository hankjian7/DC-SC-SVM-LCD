#pragma once
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

struct Item {
    int id;
    int frequency;
    int flag;
    int weight;
    
    Item(int i, int f, int fl, int w) : id(i), frequency(f), flag(fl), weight(w) {}
};

class MinHeapWithFlag {
public:
    MinHeapWithFlag() = default;

    // Modified to include flag and weight
    void push(Item &item);

    // Returns the item with the highest priority
    Item pop();

    Item peek() const;

    bool isEmpty() const;

    void incrementCount(int id, int add_count);
    
    // Set or update flag for an item
    void setFlag(int id, int flag);
    
    // Set or update weight for an item
    void setWeight(int id, int weight);

    friend std::ostream& operator<<(std::ostream& os, const MinHeapWithFlag& heap);

private:
    // Changed to store Item structure
    std::vector<Item> heap;
    std::unordered_map<int, int> id_to_index;

    void heapify(int index);

    void siftUp(int index);

    void swap(int i, int j);
    
    // Helper method to compare two items based on custom rules
    bool hasHigherPriority(const Item& a, const Item& b) const;
};