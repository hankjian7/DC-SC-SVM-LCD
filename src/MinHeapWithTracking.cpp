#include "MinHeapWithTracking.hpp"

void MinHeapWithTracking::push(const std::pair<int, int>& item) 
{
    heap.push_back(item);
    int index = heap.size() - 1;
    id_to_index[item.first] = index;
    sift_up(index);
}

std::pair<int, int> MinHeapWithTracking::pop() 
{
    if (heap.empty()) {
        throw std::out_of_range("Heap is empty");
    }
    if (heap.size() == 1) {
        auto item = heap.back();
        heap.pop_back();
        id_to_index.erase(item.first);
        return item;
    }

    auto root = heap[0];
    auto last_item = heap.back();
    heap[0] = last_item;
    id_to_index[last_item.first] = 0;
    heap.pop_back();
    id_to_index.erase(root.first);
    heapify(0);
    return root;
}

std::pair<int, int> MinHeapWithTracking::peek() const 
{
    if (heap.empty()) {
        throw std::out_of_range("Heap is empty");
    }
    return heap[0];
}

bool MinHeapWithTracking::is_empty() const 
{
    return heap.empty();
}

void MinHeapWithTracking::increment_count(int id, int add_count) 
{
    auto it = id_to_index.find(id);
    if (it != id_to_index.end()) {
        int index = it->second;
        int old_count = heap[index].second;
        int new_count = old_count + add_count;
        heap[index].second = new_count;
        if (new_count < old_count) {
            sift_up(index);
        } else {
            heapify(index);
        }
    } else {
        throw std::invalid_argument("ID not found in heap");
    }
}

void MinHeapWithTracking::heapify(int index) 
{
    int smallest = index;
    int left = 2 * index + 1;
    int right = 2 * index + 2;

    if (left < heap.size() && heap[left].second < heap[smallest].second) {
        smallest = left;
    }

    if (right < heap.size() && heap[right].second < heap[smallest].second) {
        smallest = right;
    }

    if (smallest != index) {
        swap(index, smallest);
        heapify(smallest);
    }
}

void MinHeapWithTracking::sift_up(int index) 
{
    int current = index;
    while (current > 0) {
        int parent = (current - 1) / 2;
        if (heap[current].second < heap[parent].second) {
            swap(current, parent);
            current = parent;
        } else {
            break;
        }
    }
}

void MinHeapWithTracking::swap(int i, int j) 
{
    id_to_index[heap[i].first] = j;
    id_to_index[heap[j].first] = i;
    std::swap(heap[i], heap[j]);
}

std::ostream& operator<<(std::ostream& os, const MinHeapWithTracking& heap) 
{
    for (const auto& item : heap.heap) {
        os << "(" << item.first << ", " << item.second << ") ";
    }
    return os;
}
