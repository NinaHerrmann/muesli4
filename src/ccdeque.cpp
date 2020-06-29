template <class T>
void msl::detail::CcDeque<T>::push_front(const T& val)
{
  std::lock_guard<std::mutex> lock(mtx);
	data.push_front(val);
}

template <class T>
void msl::detail::CcDeque<T>::push_back(const T& val)
{
  std::lock_guard<std::mutex> lock(mtx);
  data.push_back(val);
}

template <class T>
bool msl::detail::CcDeque<T>::try_pop_back(T& popped_value)
{
  std::lock_guard<std::mutex> lock(mtx);
	if (data.empty()) {
		return false;
	}

	popped_value = data.back();
	data.pop_back();
	return true;
}

template <class T>
bool msl::detail::CcDeque<T>::try_pop_front(T& popped_value)
{
  std::lock_guard<std::mutex> lock(mtx);
	if (data.empty()) {
		return false;
	}

	popped_value = data.front();
	data.pop_front();
	return true;
}


template <class T>
bool msl::detail::CcDeque<T>::empty() const
{
  std::lock_guard<std::mutex> lock(mtx);
	return data.empty();
}

template <class T>
int msl::detail::CcDeque<T>::size() const
{
  std::lock_guard<std::mutex> lock(mtx);
	return data.size();
}

