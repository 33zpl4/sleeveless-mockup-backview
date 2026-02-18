/*
 * Generate Back-View Mockup from Front-View Sleeveless Shirt
 *
 * Run in Photoshop: File > Scripts > Browse > select this file
 *
 * Approach:
 *   1. Duplicates the document (original untouched)
 *   2. Flips the canvas horizontally
 *   3. Creates a "stamp visible" pixel layer on top
 *   4. Patches the neckline + removes center line on that layer
 *   5. Saves as layered TIFF
 *
 * The patch layer sits on top and covers the neckline area.
 * Original layers (SHIRT group, BLANK IMAGE, etc.) stay intact below.
 */

#target photoshop

(function () {
    if (!app.documents.length) {
        alert("No document is open.\nPlease open the front-view mockup first.");
        return;
    }

    var origDoc = app.activeDocument;
    var origUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    try {
        // ---- 1. Duplicate ----
        var baseName = origDoc.name.replace(/\.[^.]+$/, "");
        var backDoc = origDoc.duplicate(baseName + "_back_view");
        app.activeDocument = backDoc;

        var W = backDoc.width.as("px");
        var H = backDoc.height.as("px");

        // ---- 2. Flip canvas horizontally ----
        backDoc.flipCanvas(Direction.HORIZONTAL);

        // ---- 3. Stamp visible to a new pixel layer ----
        // This merges everything visible into one layer we can edit,
        // while keeping all original layers intact below.
        var patchLayer = stampVisible(backDoc);
        patchLayer.name = "Back View Patch";

        // ---- 4. Detect shirt bounds for neckline estimation ----
        // Find the SHIRT group to get its bounds
        var shirtBounds = getShirtBounds(backDoc);

        var shirtLeft, shirtTop, shirtRight, shirtBottom;
        if (shirtBounds) {
            shirtLeft   = shirtBounds[0];
            shirtTop    = shirtBounds[1];
            shirtRight  = shirtBounds[2];
            shirtBottom = shirtBounds[3];
        } else {
            // Fallback: assume shirt occupies the center ~60% of canvas
            shirtLeft   = W * 0.20;
            shirtTop    = H * 0.08;
            shirtRight  = W * 0.80;
            shirtBottom = H * 0.85;
        }

        var shirtW = shirtRight - shirtLeft;
        var shirtH = shirtBottom - shirtTop;
        var centerX = (shirtLeft + shirtRight) / 2;

        // ---- 5. Patch the neckline ----
        // Front neckline dip is roughly 15-20% of shirt height.
        // We fill the lower ~60% of it to raise it for the back view.
        var neckDepth = shirtH * 0.16;
        var neckWidth = shirtW * 0.28;

        // Elliptical selection covering the neckline opening
        var neckLeft   = centerX - neckWidth / 2;
        var neckRight  = centerX + neckWidth / 2;
        var neckTop    = shirtTop;
        var neckBottom = shirtTop + neckDepth;

        // Make sure the patch layer is active
        backDoc.activeLayer = patchLayer;

        // Select and fill the neckline area
        selectEllipse(neckLeft, neckTop, neckRight, neckBottom, 20);

        try {
            contentAwareFill();
        } catch (e1) {
            // Fallback: clone from fabric area below the neckline
            try {
                fillFromSurrounding(backDoc, centerX, neckBottom + 80);
            } catch (e2) {
                // Leave for manual touch-up
            }
        }
        backDoc.selection.deselect();

        // ---- 6. Remove center guide line ----
        var lineHalfW = 20;
        var lineTop = neckBottom;
        var lineBottom = shirtBottom - shirtH * 0.05;

        var lineRegion = [
            [centerX - lineHalfW, lineTop],
            [centerX + lineHalfW, lineTop],
            [centerX + lineHalfW, lineBottom],
            [centerX - lineHalfW, lineBottom]
        ];
        backDoc.selection.select(lineRegion, SelectionType.REPLACE, 8, false);

        try {
            contentAwareFill();
        } catch (e3) {
            // Skip if fails
        }
        backDoc.selection.deselect();

        // ---- 7. Save as layered TIFF ----
        var savePath;
        try {
            savePath = origDoc.path;
        } catch (ep) {
            savePath = Folder.desktop;
        }
        var outputFile = new File(savePath + "/" + baseName + "_back_view.tif");

        var tiffOpts = new TiffSaveOptions();
        tiffOpts.layers = true;
        tiffOpts.alphaChannels = true;
        tiffOpts.imageCompression = TIFFEncoding.TIFFLZW;
        tiffOpts.layerCompression = LayerCompression.ZIP;

        backDoc.saveAs(outputFile, tiffOpts, true, Extension.LOWERCASE);

        alert("Done! Back-view mockup saved:\n" + outputFile.fsName +
              "\n\nThe 'Back View Patch' layer covers the neckline.\n" +
              "Review and touch up if needed.");

    } catch (e) {
        alert("Error at line " + e.line + ":\n" + e.message);
    } finally {
        app.preferences.rulerUnits = origUnits;
    }


    // ===========================================================
    //  HELPER FUNCTIONS
    // ===========================================================

    /**
     * Stamp Visible: creates a new layer with the merged composite.
     * Equivalent to Ctrl+Alt+Shift+E.
     * Original layers remain untouched.
     */
    function stampVisible(doc) {
        // Select topmost layer so stamp goes on top
        doc.activeLayer = doc.layers[0];

        // Create a new empty layer
        var stamp = doc.artLayers.add();
        stamp.name = "Stamp";
        doc.activeLayer = stamp;

        // Apply Image (merge visible onto this layer)
        // Using Action Manager for "Stamp Visible"
        var desc = new ActionDescriptor();
        desc.putClass(charIDToTypeID("Nw  "), charIDToTypeID("Lyr "));
        desc.putEnumerated(
            stringIDToTypeID("using"),
            stringIDToTypeID("mergeVisibleEvent"),
            stringIDToTypeID("mergeVisibleEvent")
        );

        try {
            executeAction(charIDToTypeID("Mk  "), desc, DialogModes.NO);
        } catch (e) {
            // Fallback: Select All > Copy Merged > Paste
            stamp.remove();
            doc.selection.selectAll();
            doc.selection.copy(true); // true = merge visible
            doc.paste();
            doc.activeLayer.name = "Stamp";
        }

        return doc.activeLayer;
    }

    /**
     * Find the SHIRT layer/group bounds.
     */
    function getShirtBounds(doc) {
        for (var i = 0; i < doc.layers.length; i++) {
            var name = doc.layers[i].name.toUpperCase();
            if (name === "SHIRT" || name.indexOf("SHIRT") >= 0) {
                var b = doc.layers[i].bounds;
                return [
                    b[0].as("px"),
                    b[1].as("px"),
                    b[2].as("px"),
                    b[3].as("px")
                ];
            }
        }
        return null;
    }

    /**
     * Make an elliptical selection.
     */
    function selectEllipse(left, top, right, bottom, feather) {
        var desc = new ActionDescriptor();
        var ref = new ActionReference();
        ref.putProperty(charIDToTypeID("Chnl"), charIDToTypeID("fsel"));
        desc.putReference(charIDToTypeID("null"), ref);

        var shape = new ActionDescriptor();
        shape.putUnitDouble(charIDToTypeID("Top "), charIDToTypeID("#Pxl"), top);
        shape.putUnitDouble(charIDToTypeID("Left"), charIDToTypeID("#Pxl"), left);
        shape.putUnitDouble(charIDToTypeID("Btom"), charIDToTypeID("#Pxl"), bottom);
        shape.putUnitDouble(charIDToTypeID("Rght"), charIDToTypeID("#Pxl"), right);

        desc.putObject(charIDToTypeID("T   "), charIDToTypeID("Elps"), shape);
        desc.putUnitDouble(charIDToTypeID("Fthr"), charIDToTypeID("#Pxl"), feather || 0);
        desc.putBoolean(charIDToTypeID("AntA"), true);

        executeAction(charIDToTypeID("setd"), desc, DialogModes.NO);
    }

    /**
     * Content-Aware Fill on the current selection.
     */
    function contentAwareFill() {
        var desc = new ActionDescriptor();
        desc.putEnumerated(
            charIDToTypeID("Usng"),
            charIDToTypeID("FlCn"),
            stringIDToTypeID("contentAware")
        );
        desc.putUnitDouble(charIDToTypeID("Opct"), charIDToTypeID("#Prc"), 100);
        desc.putEnumerated(
            charIDToTypeID("Md  "),
            charIDToTypeID("BlnM"),
            charIDToTypeID("Nrml")
        );
        executeAction(charIDToTypeID("Fl  "), desc, DialogModes.NO);
    }

    /**
     * Fallback fill: sample a color from the fabric and fill.
     */
    function fillFromSurrounding(doc, x, y) {
        x = Math.min(Math.max(x, 0), doc.width.as("px") - 1);
        y = Math.min(Math.max(y, 0), doc.height.as("px") - 1);

        var sampler = doc.colorSamplers.add([
            UnitValue(x, "px"),
            UnitValue(y, "px")
        ]);
        var fabricColor = new SolidColor();
        fabricColor.rgb.red   = sampler.color.rgb.red;
        fabricColor.rgb.green = sampler.color.rgb.green;
        fabricColor.rgb.blue  = sampler.color.rgb.blue;
        sampler.remove();

        doc.selection.fill(fabricColor, ColorBlendMode.NORMAL, 100, false);
    }

})();
