package com.ifmo.machinelearning.master_homework1

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import com.ifmo.machinelearning.visualization.Plot2DBuilder
import java.awt.Color
import java.io.File

/**
 * Created by warrior on 10/24/16.
 */

const val MAX_ATTRIBUTE_NUMBER = 100

fun main(args: Array<String>) {
    val mapper = ObjectMapper()
    val quality = listOf("best", "worst")
    val fullResults: Map<String, Double> = mapper.readValue(File("$CLASSIFICATION_RESULTS/results.json"), object : TypeReference<Map<String, Double>>() {})


    for (classifier in Classifier.values().map { it.name.toLowerCase() }) {
        val builder = Plot2DBuilder("attributes", "F-measure")
        val withAllAttributes = fullResults[classifier]
        if (withAllAttributes != null) {
            builder.addPlot("all", Color.BLACK,
                    DoubleArray(MAX_ATTRIBUTE_NUMBER, { i -> i + 1.0 }),
                    DoubleArray(MAX_ATTRIBUTE_NUMBER, { withAllAttributes }))
        }
        for (ranker in FeatureRanker.values().map { it.name.toLowerCase() }) {
            for (q in quality) {
                val fileName = "$CLASSIFICATION_RESULTS/$classifier/$classifier-$ranker-$q.json"
                val results = mapper.readValue(File(fileName), Result::class.java).bestAttributes
                val bests = results.take(MAX_ATTRIBUTE_NUMBER).toDoubleArray()
                builder.addPlot("$ranker-$q", DoubleArray(bests.size, { i -> i + 1.0 }), bests)
            }
        }
        builder.show(classifier, 1200, 600)
    }
}
