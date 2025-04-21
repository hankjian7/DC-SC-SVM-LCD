#include "MinHeapWithFlag.hpp"

void MinHeapWithFlag::push(Item &item) 
{
    heap.push_back(item);
    int index = heap.size() - 1;
    id_to_index[item.id] = index;
    siftUp(index);
}

Item MinHeapWithFlag::pop() 
{
    if (heap.empty()) {
        throw std::out_of_range("Heap is empty");
    }
    if (heap.size() == 1) {
        auto item = heap.back();
        heap.pop_back();
        id_to_index.erase(item.id);
        return item;
    }

    auto root = heap[0];
    auto last_item = heap.back();
    heap[0] = last_item;
    id_to_index[last_item.id] = 0;
    heap.pop_back();
    id_to_index.erase(root.id);
    heapify(0);
    return root;
}

Item MinHeapWithFlag::peek() const 
{
    if (heap.empty()) {
        throw std::out_of_range("Heap is empty");
    }
    return heap[0];
}

bool MinHeapWithFlag::isEmpty() const 
{
    return heap.empty();
}

void MinHeapWithFlag::incrementCount(int id, int add_count) 
{
    auto it = id_to_index.find(id);
    if (it != id_to_index.end()) {
        int index = it->second;
        int old_count = heap[index].frequency;
        int new_count = old_count + add_count;
        
        // Update frequency
        heap[index].frequency = new_count;
        
        if (new_count < old_count) {
            siftUp(index);
        } else {
            heapify(index);
        }
    } else {
        throw std::invalid_argument("ID not found in heap");
    }
}

void MinHeapWithFlag::setFlag(int id, int flag) 
{
    auto it = id_to_index.find(id);
    if (it != id_to_index.end()) {
        int index = it->second;
        int old_flag = heap[index].flag;
        
        // Update flag
        heap[index].flag = flag;
        
        // If flag changed to 1 (higher priority), sift up
        if (flag > old_flag) {
            siftUp(index);
        } else if (flag < old_flag) {
            heapify(index);
        }
    } else {
        throw std::invalid_argument("ID not found in heap");
    }
}

void MinHeapWithFlag::setWeight(int id, int weight) 
{
    auto it = id_to_index.find(id);
    if (it != id_to_index.end()) {
        int index = it->second;
        int old_weight = heap[index].weight;
        
        // Update weight
        heap[index].weight = weight;
        
        // If weight decreased (higher priority), sift up
        if (weight < old_weight) {
            siftUp(index);
        } else if (weight > old_weight) {
            heapify(index);
        }
    } else {
        throw std::invalid_argument("ID not found in heap");
    }
}

bool MinHeapWithFlag::hasHigherPriority(const Item& a, const Item& b) const 
{
    // First, compare by frequency (smaller is higher priority)
    if (a.frequency != b.frequency) {
        return a.frequency < b.frequency;
    }
    
    // If frequencies are equal, compare by flag (1 is higher priority)
    if (a.flag != b.flag) {
        return a.flag > b.flag;
    }
    
    // If flags are also equal, compare by weight (smaller is higher priority)
    return a.weight < b.weight;
}

void MinHeapWithFlag::heapify(int index) 
{
    int smallest = index;
    int left = 2 * index + 1;
    int right = 2 * index + 2;

    // Use the custom comparison function
    if (left < heap.size() && hasHigherPriority(heap[left], heap[smallest])) {
        smallest = left;
    }

    if (right < heap.size() && hasHigherPriority(heap[right], heap[smallest])) {
        smallest = right;
    }

    if (smallest != index) {
        swap(index, smallest);
        heapify(smallest);
    }
}

void MinHeapWithFlag::siftUp(int index) 
{
    int current = index;
    while (current > 0) {
        int parent = (current - 1) / 2;
        // Use the custom comparison function
        if (hasHigherPriority(heap[current], heap[parent])) {
            swap(current, parent);
            current = parent;
        } else {
            break;
        }
    }
}

void MinHeapWithFlag::swap(int i, int j) 
{
    id_to_index[heap[i].id] = j;
    id_to_index[heap[j].id] = i;
    std::swap(heap[i], heap[j]);
}

std::ostream& operator<<(std::ostream& os, const MinHeapWithFlag& heap) 
{
    for (const auto& item : heap.heap) {
        os << "(" << item.id << ", " 
           << item.frequency << ", " 
           << item.flag << ", " 
           << item.weight << ") ";
    }
    return os;
}