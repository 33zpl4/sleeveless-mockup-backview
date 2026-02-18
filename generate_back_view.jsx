/*
 * Generate Back-View Mockup from Front-View Sleeveless Shirt
 *
 * Run in Photoshop: File > Scripts > Browse > select this file
 *
 * What it does:
 *   1. Duplicates the document (preserves the original)
 *   2. Flips the canvas horizontally
 *   3. Raises the neckline by expanding the SHIRT group's layer mask
 *   4. Uses Content-Aware Fill to generate fabric in the neckline area
 *   5. Removes the front center guide line
 *   6. Saves as a new layered TIFF
 */

#target photoshop

(function () {
    // ---- Safety check ----
    if (!app.documents.length) {
        alert("No document is open. Please open the front-view mockup first.");
        return;
    }

    // ---- Settings ----
    var NECKLINE_RAISE = 0.55;   // How much of the neckline depth to fill (0-1)
    var CENTER_LINE_WIDTH = 40;  // Pixel width to clean the green center line

    var origDoc = app.activeDocument;
    var origUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    try {
        // ---- Step 1: Duplicate the document ----
        var baseName = origDoc.name.replace(/\.[^.]+$/, "");
        var backDoc = origDoc.duplicate(baseName + "_back_view");
        app.activeDocument = backDoc;

        var W = backDoc.width.as("px");
        var H = backDoc.height.as("px");

        // ---- Step 2: Flip canvas horizontally ----
        backDoc.flipCanvas(Direction.HORIZONTAL);

        // ---- Step 3: Find the SHIRT layer/group ----
        var shirtLayer = findLayer(backDoc, "SHIRT");
        if (!shirtLayer) {
            // Try case-insensitive search
            shirtLayer = findLayerCaseInsensitive(backDoc, "shirt");
        }

        if (shirtLayer) {
            // ---- Step 4: Raise the neckline ----
            raiseNeckline(backDoc, shirtLayer, W, H, NECKLINE_RAISE);
        } else {
            alert("Could not find 'SHIRT' layer/group.\n" +
                  "The canvas has been flipped. You may need to adjust the neckline manually.");
        }

        // ---- Step 5: Remove center guide line ----
        removeCenterLine(backDoc, W, H, CENTER_LINE_WIDTH);

        // ---- Step 6: Save as layered TIFF ----
        var outputPath = origDoc.path;
        var outputFile = new File(outputPath + "/" + baseName + "_back_view.tif");

        var tiffOpts = new TiffSaveOptions();
        tiffOpts.layers = true;
        tiffOpts.alphaChannels = true;
        tiffOpts.imageCompression = TIFFEncoding.TIFFLZW;
        tiffOpts.layerCompression = LayerCompression.ZIP;

        backDoc.saveAs(outputFile, tiffOpts, true, Extension.LOWERCASE);

        alert("Back-view mockup saved to:\n" + outputFile.fsName +
              "\n\nReview the neckline area and touch up if needed.");

    } catch (e) {
        alert("Error: " + e.message + "\nLine: " + e.line);
    } finally {
        app.preferences.rulerUnits = origUnits;
    }


    // ===========================================================
    //  HELPER FUNCTIONS
    // ===========================================================

    /**
     * Find a layer by exact name (searches top-level layers only).
     */
    function findLayer(doc, name) {
        for (var i = 0; i < doc.layers.length; i++) {
            if (doc.layers[i].name === name) {
                return doc.layers[i];
            }
        }
        return null;
    }

    /**
     * Find a layer by case-insensitive name (searches top-level + one level deep).
     */
    function findLayerCaseInsensitive(doc, name) {
        var target = name.toLowerCase();
        for (var i = 0; i < doc.layers.length; i++) {
            if (doc.layers[i].name.toLowerCase() === target) {
                return doc.layers[i];
            }
            // Search inside layer groups
            if (doc.layers[i].typename === "LayerSet") {
                var group = doc.layers[i];
                for (var j = 0; j < group.layers.length; j++) {
                    if (group.layers[j].name.toLowerCase() === target) {
                        return group.layers[j];
                    }
                }
            }
        }
        return null;
    }

    /**
     * Raise the neckline by expanding the SHIRT group's layer mask
     * and filling the exposed area with Content-Aware Fill.
     */
    function raiseNeckline(doc, shirtLayer, W, H, fillRatio) {

        // First, flatten a temp copy to detect the neckline position
        // We'll use the shirt layer's bounds to estimate the neckline area

        var bounds = shirtLayer.bounds;
        // bounds = [left, top, right, bottom] as UnitValue
        var shirtLeft = bounds[0].as("px");
        var shirtTop = bounds[1].as("px");
        var shirtRight = bounds[2].as("px");
        var shirtBottom = bounds[3].as("px");

        var shirtW = shirtRight - shirtLeft;
        var shirtH = shirtBottom - shirtTop;

        // Neckline is roughly in the top-center of the shirt
        // For a sleeveless shirt, the neckline dip is typically 10-20% of shirt height
        var neckDepthEstimate = shirtH * 0.12;
        var neckWidthEstimate = shirtW * 0.30;

        var neckCenterX = (shirtLeft + shirtRight) / 2;
        var neckTopY = shirtTop;
        var neckBottomY = shirtTop + neckDepthEstimate;

        // The area to fill: an elliptical region covering the neckline dip
        // We'll raise it by fillRatio
        var fillTopY = neckTopY + neckDepthEstimate * (1 - fillRatio);
        var fillBottomY = neckBottomY;

        var fillLeft = neckCenterX - neckWidthEstimate / 2;
        var fillRight = neckCenterX + neckWidthEstimate / 2;

        // ---- Expand the layer mask in the neckline area ----
        if (hasLayerMask(shirtLayer)) {
            // Select the neckline fill region (elliptical)
            var selBounds = [
                [fillLeft, fillTopY],
                [fillRight, fillTopY],
                [fillRight, fillBottomY],
                [fillLeft, fillBottomY]
            ];
            doc.selection.select(selBounds, SelectionType.REPLACE, 20, false);

            // Make the elliptical by selecting ellipse
            selectEllipse(doc, fillLeft, fillTopY, fillRight, fillBottomY, 10);

            // Switch to the layer mask and fill with white (reveal)
            selectLayerMask();
            var white = new SolidColor();
            white.rgb.red = 255;
            white.rgb.green = 255;
            white.rgb.blue = 255;
            doc.selection.fill(white, ColorBlendMode.NORMAL, 100, false);

            // Switch back to the layer content
            selectLayerContent();
            doc.selection.deselect();
        }

        // ---- Content-Aware Fill on the neckline area ----
        // Select the neckline region on the composite
        selectEllipse(doc, fillLeft, fillTopY, fillRight, fillBottomY, 15);

        // Merge visible to a temp layer for content-aware fill
        try {
            contentAwareFill();
        } catch (e) {
            // Content-Aware Fill may fail in some PS versions; try regular fill
            try {
                var fabricColor = sampleFabricColor(doc, shirtLeft, shirtRight, neckBottomY);
                doc.selection.fill(fabricColor, ColorBlendMode.NORMAL, 100, false);
            } catch (e2) {
                // If all else fails, just leave it for manual touch-up
            }
        }

        doc.selection.deselect();
    }

    /**
     * Select an elliptical region.
     */
    function selectEllipse(doc, left, top, right, bottom, feather) {
        var desc = new ActionDescriptor();
        var ref = new ActionReference();
        ref.putProperty(charIDToTypeID("Chnl"), charIDToTypeID("fsel"));
        desc.putReference(charIDToTypeID("null"), ref);

        var ellipseDesc = new ActionDescriptor();
        ellipseDesc.putUnitDouble(charIDToTypeID("Top "), charIDToTypeID("#Pxl"), top);
        ellipseDesc.putUnitDouble(charIDToTypeID("Left"), charIDToTypeID("#Pxl"), left);
        ellipseDesc.putUnitDouble(charIDToTypeID("Btom"), charIDToTypeID("#Pxl"), bottom);
        ellipseDesc.putUnitDouble(charIDToTypeID("Rght"), charIDToTypeID("#Pxl"), right);

        desc.putObject(charIDToTypeID("T   "), charIDToTypeID("Elps"), ellipseDesc);
        desc.putUnitDouble(charIDToTypeID("Fthr"), charIDToTypeID("#Pxl"), feather || 0);
        desc.putBoolean(charIDToTypeID("AntA"), true);

        executeAction(charIDToTypeID("setd"), desc, DialogModes.NO);
    }

    /**
     * Check if a layer has a layer mask.
     */
    function hasLayerMask(layer) {
        try {
            var ref = new ActionReference();
            ref.putEnumerated(charIDToTypeID("Lyr "), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
            var desc = executeActionGet(ref);
            return desc.hasKey(charIDToTypeID("UsrM"));
        } catch (e) {
            return false;
        }
    }

    /**
     * Select (target) the layer mask for editing.
     */
    function selectLayerMask() {
        var desc = new ActionDescriptor();
        var ref = new ActionReference();
        ref.putEnumerated(charIDToTypeID("Chnl"), charIDToTypeID("Chnl"), charIDToTypeID("Msk "));
        desc.putReference(charIDToTypeID("null"), ref);
        desc.putBoolean(charIDToTypeID("MkVs"), false);
        executeAction(charIDToTypeID("slct"), desc, DialogModes.NO);
    }

    /**
     * Switch back to editing the layer content (RGB).
     */
    function selectLayerContent() {
        var desc = new ActionDescriptor();
        var ref = new ActionReference();
        ref.putEnumerated(charIDToTypeID("Chnl"), charIDToTypeID("Chnl"), charIDToTypeID("RGB "));
        desc.putReference(charIDToTypeID("null"), ref);
        executeAction(charIDToTypeID("slct"), desc, DialogModes.NO);
    }

    /**
     * Execute Content-Aware Fill on the current selection.
     */
    function contentAwareFill() {
        var desc = new ActionDescriptor();
        desc.putEnumerated(
            charIDToTypeID("Usng"),
            charIDToTypeID("FlCn"),
            stringIDToTypeID("contentAware")
        );
        desc.putUnitDouble(charIDToTypeID("Opct"), charIDToTypeID("#Prc"), 100);
        desc.putEnumerated(charIDToTypeID("Md  "), charIDToTypeID("BlnM"), charIDToTypeID("Nrml"));
        executeAction(charIDToTypeID("Fl  "), desc, DialogModes.NO);
    }

    /**
     * Sample the average fabric color from beside the neckline.
     */
    function sampleFabricColor(doc, shirtLeft, shirtRight, belowNeckY) {
        var sampleX = (shirtLeft + shirtRight) / 2;
        var sampleY = belowNeckY + 50;

        // Clamp to document bounds
        sampleX = Math.min(Math.max(sampleX, 0), doc.width.as("px") - 1);
        sampleY = Math.min(Math.max(sampleY, 0), doc.height.as("px") - 1);

        return doc.colorSamplers.add([UnitValue(sampleX, "px"), UnitValue(sampleY, "px")]).color;
    }

    /**
     * Remove the vertical center guide line using Content-Aware Fill.
     */
    function removeCenterLine(doc, W, H, lineWidth) {
        var centerX = W / 2;
        var halfW = lineWidth / 2;

        // Select a thin vertical strip at the center
        // Only in the shirt area (roughly top 15% to 70% of image)
        var selTop = H * 0.10;
        var selBottom = H * 0.75;
        var selLeft = centerX - halfW;
        var selRight = centerX + halfW;

        var region = [
            [selLeft, selTop],
            [selRight, selTop],
            [selRight, selBottom],
            [selLeft, selBottom]
        ];
        doc.selection.select(region, SelectionType.REPLACE, 5, false);

        // Try Content-Aware Fill on the center line
        try {
            contentAwareFill();
        } catch (e) {
            // If it fails, skip - user can manually fix
        }

        doc.selection.deselect();
    }

})();
