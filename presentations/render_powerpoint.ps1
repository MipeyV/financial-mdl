param(
    [Parameter(Mandatory = $true)][string]$PresentationPath,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $PresentationPath).Path
$renderPath = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $renderPath) {
    throw "Use a fresh render directory: $renderPath"
}
New-Item -ItemType Directory -Path $renderPath -Force | Out-Null
$application = $null
$presentation = $null
$probe = $null
$slides = @()
$overflows = @()
$cells = @()
try {
    $application = New-Object -ComObject PowerPoint.Application
    # Open without a presentation window. Existing user presentations remain untouched.
    $presentation = $application.Presentations.Open($sourcePath, -1, 0, 0)
    for ($index = 1; $index -le $presentation.Slides.Count; $index++) {
        $slide = $presentation.Slides.Item($index)
        $imagePath = Join-Path $renderPath ('slide-{0:D2}.png' -f $index)
        $slide.Export($imagePath, 'PNG', 1600, 900)
        $chartCount = 0
        $tableCount = 0
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasChart -eq -1) { $chartCount++ }
            if ($shape.HasTable -eq -1) {
                $tableCount++
                for ($row = 1; $row -le $shape.Table.Rows.Count; $row++) {
                    for ($column = 1; $column -le $shape.Table.Columns.Count; $column++) {
                        $cellShape = $shape.Table.Cell($row, $column).Shape
                        $cells += [pscustomobject]@{
                            slide = $index; row = $row; column = $column
                            text = $cellShape.TextFrame2.TextRange.Text
                            left = $cellShape.Left; width = $cellShape.Width
                            text_left = $cellShape.TextFrame2.TextRange.BoundLeft
                            text_width = $cellShape.TextFrame2.TextRange.BoundWidth
                        }
                    }
                }
            }
            if (($shape.HasTextFrame -eq -1) -and ($shape.TextFrame.HasText -eq -1)) {
                $range = $shape.TextFrame2.TextRange
                $availableHeight = $shape.Height - $shape.TextFrame2.MarginTop - $shape.TextFrame2.MarginBottom
                if ($range.BoundHeight -gt ($availableHeight + 2)) {
                    $overflows += [pscustomobject]@{
                        slide = $index; shape = $shape.Id; text = $range.Text
                        bound_height = $range.BoundHeight; available_height = $availableHeight
                    }
                }
            }
        }
        $slides += [pscustomobject]@{
            slide = $index; hidden = ($slide.SlideShowTransition.Hidden -eq -1)
            charts = $chartCount; tables = $tableCount; png = $imagePath
        }
    }
    $probePath = Join-Path $renderPath 'editable-chart-probe.pptx'
    $presentation.SaveCopyAs($probePath)
    $presentation.Close()
    $presentation = $null
    $probe = $application.Presentations.Open($probePath, 0, 0, 0)
    $chartShape = @($probe.Slides.Item(4).Shapes | Where-Object { $_.HasChart -eq -1 })[0]
    $originalWidth = $chartShape.Width
    $chartShape.Width = $originalWidth + 3
    $chartShape.Chart.SeriesCollection(1).Name = 'Editable chart check'
    $probe.Save()
    $probe.Close()
    $probe = $application.Presentations.Open($probePath, 0, 0, 0)
    $chartShape = @($probe.Slides.Item(4).Shapes | Where-Object { $_.HasChart -eq -1 })[0]
    if ($chartShape.Chart.SeriesCollection(1).Name -ne 'Editable chart check') {
        throw 'Native chart edit did not survive a save/reopen roundtrip.'
    }
    $probe.Close()
    $probe = $null
    [pscustomobject]@{
        application = 'Microsoft PowerPoint'; version = $application.Version
        source = $sourcePath; slides = $slides; text_overflows = $overflows; table_cells = $cells
        native_chart_edit_save_reopen = $true
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $renderPath 'powerpoint-review.json') -Encoding UTF8
    Write-Output "Rendered $($slides.Count) slides; overflow candidates: $($overflows.Count)"
}
catch {
    Write-Output $_.ScriptStackTrace
    Write-Output $_.Exception.ToString()
    throw
}
finally {
    if ($null -ne $probe) { $probe.Close() }
    if ($null -ne $presentation) { $presentation.Close() }
    if (($null -ne $application) -and ($application.Presentations.Count -eq 0)) {
        $application.Quit()
    }
    if ($null -ne $application) {
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($application) | Out-Null
    }
}
