package com.ifmo.machinelearning;

/**
 * Created by Whiplash on 19.09.2014.
 */
public interface Weight<T, E> {

    public E weight(T first, T second);

}
